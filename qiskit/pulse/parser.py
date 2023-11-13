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

"""Parser for mathematical string expressions returned by backends."""
from typing import Dict, List, Union
import ast
import copy
import operator

import cmath

from qiskit.pulse.exceptions import PulseError
from qiskit.circuit import ParameterExpression


class PulseExpression(ast.NodeTransformer):
    """Expression parser to evaluate parameter values."""

    _math_ops = {
        "acos": cmath.acos,
        "acosh": cmath.acosh,
        "asin": cmath.asin,
        "asinh": cmath.asinh,
        "atan": cmath.atan,
        "atanh": cmath.atanh,
        "cos": cmath.cos,
        "cosh": cmath.cosh,
        "exp": cmath.exp,
        "log": cmath.log,
        "log10": cmath.log10,
        "sin": cmath.sin,
        "sinh": cmath.sinh,
        "sqrt": cmath.sqrt,
        "tan": cmath.tan,
        "tanh": cmath.tanh,
        "pi": cmath.pi,
        "e": cmath.e,
    }
    """Valid math functions."""

    _binary_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }
    """Valid binary operations."""

    _unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}
    """Valid unary operations."""

    def __init__(self, source: Union[str, ast.Expression], partial_binding: bool = False):
        """Create new evaluator.

        Args:
            source: Expression of equation to evaluate.
            partial_binding: Allow partial bind of parameters.

        Raises:
            PulseError: When invalid string is specified.
        """
        self._partial_binding = partial_binding
        self._locals_dict = {}
        self._params = set()

        if isinstance(source, ast.Expression):
            self._tree = source
        else:
            try:
                self._tree = ast.parse(source, mode="eval")
            except SyntaxError as ex:
                raise PulseError(f"{source} is invalid expression.") from ex

        # parse parameters
        self.visit(self._tree)

    @property
    def params(self) -> List[str]:
        """Get parameters.

        Returns:
            A list of parameters in sorted order.
        """
        return sorted(self._params.copy())

    def __call__(self, *args, **kwargs) -> Union[complex, ast.Expression]:
        """Evaluate the expression with the given values of the expression's parameters.

        Args:
            *args: Variable length parameter list.
            **kwargs: Arbitrary parameters.

        Returns:
            Evaluated value.

        Raises:
            PulseError: When parameters are not bound.
        """
        if isinstance(self._tree.body, ast.Constant):
            return self._tree.body.value

        self._locals_dict.clear()
        if args:
            for key, val in zip(self.params, args):
                self._locals_dict[key] = val
        if kwargs:
            for key, val in kwargs.items():
                if key in self.params:
                    if key not in self._locals_dict.keys():
                        self._locals_dict[key] = val
                    else:
                        raise PulseError(
                            "%s got multiple values for argument '%s'"
                            % (self.__class__.__name__, key)
                        )
                else:
                    raise PulseError(
                        "%s got an unexpected keyword argument '%s'"
                        % (self.__class__.__name__, key)
                    )

        expr = self.visit(self._tree)

        if not isinstance(expr.body, ast.Constant):
            if self._partial_binding:
                return PulseExpression(expr, self._partial_binding)
            else:
                raise PulseError("Parameters %s are not all bound." % self.params)
        return expr.body.value

    @staticmethod
    def _match_ops(opr: ast.AST, opr_dict: Dict, *args) -> complex:
        """Helper method to apply operators.

        Args:
            opr: Operator of node.
            opr_dict: Mapper from ast to operator.
            *args: Arguments supplied to operator.

        Returns:
            Evaluated value.

        Raises:
            PulseError: When unsupported operation is specified.
        """
        for op_type, op_func in opr_dict.items():
            if isinstance(opr, op_type):
                return op_func(*args)
        raise PulseError("Operator %s is not supported." % opr.__class__.__name__)

    def visit_Expression(self, node: ast.Expression) -> ast.Expression:
        """Evaluate children nodes of expression.

        Args:
            node: Expression to evaluate.

        Returns:
            Evaluated value.
        """
        tmp_node = copy.copy(node)
        tmp_node.body = self.visit(tmp_node.body)

        return tmp_node

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        """Return constant value as it is.

        Args:
            node: Constant.

        Returns:
            Input node.
        """
        return node

    def visit_Name(self, node: ast.Name) -> Union[ast.Name, ast.Constant]:
        """Evaluate name and return ast.Constant if it is bound.

        Args:
            node: Name to evaluate.

        Returns:
            Evaluated value.

        Raises:
            PulseError: When parameter value is not a number.
        """
        if node.id in self._math_ops:
            val = ast.Constant(self._math_ops[node.id])
            return ast.copy_location(val, node)
        elif node.id in self._locals_dict:
            _val = self._locals_dict[node.id]
            if not isinstance(_val, ParameterExpression):
                # check value type
                try:
                    _val = complex(_val)
                    if not _val.imag:
                        _val = _val.real
                except ValueError as ex:
                    raise PulseError(
                        f"Invalid parameter value {node.id} = {self._locals_dict[node.id]} is "
                        "specified."
                    ) from ex
            val = ast.Constant(_val)
            return ast.copy_location(val, node)
        self._params.add(node.id)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Union[ast.UnaryOp, ast.Constant]:
        """Evaluate unary operation and return ast.Constant if operand is bound.

        Args:
            node: Unary operation to evaluate.

        Returns:
            Evaluated value.
        """
        node = copy.copy(node)
        node.operand = self.visit(node.operand)
        if isinstance(node.operand, ast.Constant):
            val = ast.Constant(self._match_ops(node.op, self._unary_ops, node.operand.value))
            return ast.copy_location(val, node)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> Union[ast.BinOp, ast.Constant]:
        """Evaluate binary operation and return ast.Constant if operands are bound.

        Args:
            node: Binary operation to evaluate.

        Returns:
            Evaluated value.
        """
        node = copy.copy(node)
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            val = ast.Constant(
                self._match_ops(node.op, self._binary_ops, node.left.value, node.right.value)
            )
            return ast.copy_location(val, node)
        return node

    def visit_Call(self, node: ast.Call) -> Union[ast.Call, ast.Constant]:
        """Evaluate function and return ast.Constant if all arguments are bound.

        Args:
            node: Function to evaluate.

        Returns:
            Evaluated value.

        Raises:
            PulseError: When unsupported or unsafe function is specified.
        """
        if not isinstance(node.func, ast.Name):
            raise PulseError("Unsafe expression is detected.")
        node = copy.copy(node)
        node.args = [self.visit(arg) for arg in node.args]
        if all(isinstance(arg, ast.Constant) for arg in node.args):
            if node.func.id not in self._math_ops.keys():
                raise PulseError("Function %s is not supported." % node.func.id)
            _args = [arg.value for arg in node.args]
            _val = self._math_ops[node.func.id](*_args)
            if not _val.imag:
                _val = _val.real
            val = ast.Constant(_val)
            return ast.copy_location(val, node)
        return node

    def generic_visit(self, node):
        raise PulseError("Unsupported node: %s" % node.__class__.__name__)


def parse_string_expr(source: str, partial_binding: bool = False):
    """Safe parsing of string expression.

    Args:
        source: String expression to parse.
        partial_binding: Allow partial bind of parameters.

    Returns:
        PulseExpression: Returns a expression object.

    Example:

        expr = 'P1 + P2 + P3'
        parsed_expr = parse_string_expr(expr, partial_binding=True)

        # create new PulseExpression
        bound_two = parsed_expr(P1=1, P2=2)
        # evaluate expression
        value1 = bound_two(P3=3)
        value2 = bound_two(P3=4)
        value3 = bound_two(P3=5)

    """
    subs = [("numpy.", ""), ("np.", ""), ("math.", ""), ("cmath.", "")]
    for match, sub in subs:
        source = source.replace(match, sub)

    return PulseExpression(source, partial_binding)
