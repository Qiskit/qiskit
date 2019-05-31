# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Helper function to parse string expression given by backends."""

import ast
import copy
import operator
import functools

import cmath

from qiskit.exceptions import QiskitError


class PulseExpression(ast.NodeTransformer):
    """Expression parser to evaluate parameter values.
    """
    # valid functions
    _math_ops = {
        'acos': cmath.acos,
        'acosh': cmath.acosh,
        'asin': cmath.asin,
        'asinh': cmath.asinh,
        'atan': cmath.atan,
        'atanh': cmath.atanh,
        'cos': cmath.cos,
        'cosh': cmath.cosh,
        'exp': cmath.exp,
        'log': cmath.log,
        'log10': cmath.log10,
        'sin': cmath.sin,
        'sinh': cmath.sinh,
        'sqrt': cmath.sqrt,
        'tan': cmath.tan,
        'tanh': cmath.tanh,
        'pi': cmath.pi,
        'e': cmath.e
    }

    # valid binary operations
    _binary_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow
    }

    # valid unary operations
    _unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg
    }

    def __init__(self, source, partial_binding=False):
        """Create new evaluator.

        Args:
            source (str): String expression of equation to evaluate.
            partial_binding (bool): Allow partial bind of parameters.
        """
        self._partial_binding = partial_binding
        self._locals_dict = {}
        self._params = set()
        self._tree = None

        try:
            self._tree = ast.parse(source, mode='eval')
        except SyntaxError:
            raise QiskitError('%s is invalid expression.' % source)

        self.visit(self._tree)

    @property
    def params(self):
        """Get parameters."""
        return sorted(self._params.copy())

    def __call__(self, *args, **kwargs):
        """Get evaluated value with given parameters.

        Args:
            *args: Variable length parameter list.
            **kwargs: Arbitrary parameters.

        Returns:
            float or complex or ast: Evaluated value.

        Raises:
            QiskitError: When parameters are not bound.
        """
        self._locals_dict = {}

        if args:
            for key, val in zip(self.params, args):
                self._locals_dict[key] = val
        if kwargs:
            self._locals_dict.update({key: val for key, val in
                                      kwargs.items() if key in self.params})

        expr = self.visit(copy.deepcopy(self._tree))

        if not isinstance(expr, ast.Num):
            if self._partial_binding:
                partial_bind = functools.partial(self.__call__, **self._locals_dict)
                return partial_bind
            else:
                raise QiskitError('Parameters %s are not all bound.' % self.params)
        return expr.n

    @staticmethod
    def _match_ops(opr, opr_dict, *args):
        """Helper method to apply operators.

        Args:
            opr (ast.AST): Operator of node.
            opr_dict (dict): Mapper from ast to operator.
            *args: Arguments supplied to operator.

        Returns:
            float or complex: Evaluated value.

        Raises:
            QiskitError: When unsupported operation is specified.
        """
        for op_type, op_func in opr_dict.items():
            if isinstance(opr, op_type):
                return op_func(*args)
        raise QiskitError('Operator %s is not supported.' % opr.__class__.__name__)

    def visit_Expression(self, node):
        """Evaluate children nodes of expression.

        Args:
            node (ast.Expression): Expression to evaluate.

        Returns:
            ast.Expression: Evaluated value.
        """
        return self.visit(node.body)

    def visit_Num(self, node):
        """Return number as it is.

        Args:
            node (ast.Num): Number.

        Returns:
            ast.Num: Number to return.
        """
        return node

    def visit_Name(self, node):
        """Evaluate name and return ast.Num if it is bound.

        Args:
            node (ast.Name): Name to evaluate.

        Returns:
            ast.Name or ast.Num: Evaluated value.

        Raises:
            QiskitError: When parameter value is not a number.
        """
        if node.id in self._math_ops.keys():
            val = ast.Num(n=self._math_ops[node.id])
            return ast.copy_location(val, node)
        elif node.id in self._locals_dict.keys():
            _val = self._locals_dict[node.id]
            try:
                _val = complex(_val)
                if not _val.imag:
                    _val = _val.real
            except ValueError:
                raise QiskitError('Invalid parameter value %s = %s is specified.'
                                  % (node.id, self._locals_dict[node.id]))
            val = ast.Num(n=_val)
            return ast.copy_location(val, node)
        self._params.add(node.id)
        return node

    def visit_UnaryOp(self, node):
        """Evaluate unary operation and return ast.Num if operand is bound.

        Args:
            node (ast.UnaryOp): Unary operation to evaluate.

        Returns:
            ast.UnaryOp or ast.Num: Evaluated value.
        """
        node.operand = self.visit(node.operand)
        if isinstance(node.operand, ast.Num):
            val = ast.Num(n=self._match_ops(node.op, self._unary_ops,
                                            node.operand.n))
            return ast.copy_location(val, node)
        return node

    def visit_BinOp(self, node):
        """Evaluate binary operation and return ast.Num if operands are bound.

        Args:
            node (ast.BinOp): Binary operation to evaluate.

        Returns:
            ast.BinOp or ast.Num: Evaluated value.
        """
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        if isinstance(node.left, ast.Num) and isinstance(node.right, ast.Num):
            val = ast.Num(n=self._match_ops(node.op, self._binary_ops,
                                            node.left.n, node.right.n))
            return ast.copy_location(val, node)
        return node

    def visit_Call(self, node):
        """Evaluate function and return ast.Num if all arguments are bound.

        Args:
            node (ast.Call): Function to evaluate.

        Returns:
            ast.Call or ast.Num: Evaluated value.

        Raises:
            QiskitError: When unsupported or unsafe function is specified.
        """
        if not isinstance(node.func, ast.Name):
            raise QiskitError('Unsafe expression is detected.')
        node.args = [self.visit(arg) for arg in node.args]
        if all(isinstance(arg, ast.Num) for arg in node.args):
            if node.func.id not in self._math_ops.keys():
                raise QiskitError('Function %s is not supported.' % node.func.id)
            _args = [arg.n for arg in node.args]
            _val = self._math_ops[node.func.id](*_args)
            if not _val.imag:
                _val = _val.real
            val = ast.Num(n=_val)
            return ast.copy_location(val, node)
        return node

    def generic_visit(self, node):
        raise QiskitError('Unsupported node: %s' % node.__class__.__name__)


def parse_string_expr(source):
    """Safe parsing of string expression.

    Args:
        source (str): String expression to parse.

    Returns:
        PulseExpression: Returns a expression object.
    """
    subs = [('numpy.', ''), ('np.', ''), ('math.', ''), ('cmath.', '')]
    for match, sub in subs:
        source = source.replace(match, sub)

    expression = PulseExpression(source)

    return expression, expression.params
