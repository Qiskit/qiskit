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
import operator
import re

import cmath

from qiskit.exceptions import QiskitError


# valid functions
_math_ops = {key: value for (key, value) in vars(cmath).items() if key[0] != '_'}

# valid parameters
_param_regex = r'(P\d+)'


def _eval_expr(parsed_expr, locals_dict):
    """ Evaluate expression without python built-in eval function.

    Args:
        parsed_expr (ast.Expression): Parsed expression to evaluate.
        locals_dict (dict): Dictionary of user arguments.

    Returns:
        complex: Evaluated value with given arguments.

    Raises:
        QiskitError: If expression is not in proper format.
    """

    def _visit_node(node):
        if isinstance(node, ast.UnaryOp):
            # Unary operation
            operand = _visit_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                opr = operator.add
            elif isinstance(node.op, ast.USub):
                opr = operator.sub
            else:
                raise QiskitError('Operator %s is not supported.' % node.op)
            return opr(0, operand)
        elif isinstance(node, ast.BinOp):
            # Binary operation
            l_value = _visit_node(node.left)
            r_value = _visit_node(node.right)
            if isinstance(node.op, ast.Add):
                opr = operator.add
            elif isinstance(node.op, ast.Sub):
                opr = operator.sub
            elif isinstance(node.op, ast.Mult):
                opr = operator.mul
            elif isinstance(node.op, ast.Div):
                opr = operator.truediv
            elif isinstance(node.op, ast.Pow):
                opr = operator.pow
            else:
                raise QiskitError('Operator %s is not supported.' % node.op)
            return opr(l_value, r_value)
        elif isinstance(node, ast.Num):
            # number
            try:
                return complex(node.n)
            except ValueError:
                raise QiskitError('Value %s cannot be converted to complex.' % node.n)
        elif isinstance(node, ast.Name):
            # parameter/constant
            if node.id in _math_ops.keys():
                return complex(_math_ops[node.id])
            else:
                # user defined args/kwargs should be numbers
                try:
                    return complex(locals_dict[node.id])
                except (KeyError, ValueError):
                    raise QiskitError('Invalid name %s is specified.' % node.id)
        elif isinstance(node, ast.Call):
            # function
            if not isinstance(node.func, ast.Name):
                raise QiskitError('Unsafe expression %s is specified.' % node.func)
            call_args = [_visit_node(arg) for arg in node.args]
            try:
                return _math_ops[node.func.id](*call_args)
            except KeyError:
                raise QiskitError('Function %s is not supported.' % node.func.id)
        else:
            raise QiskitError('Expression is written in invalid format.')

    if not isinstance(parsed_expr, ast.Expression):
        raise QiskitError('Given expression %s is not valid object.' % parsed_expr)

    return _visit_node(parsed_expr.body)


def parse_string_expr(source):
    """Safe parsing of string expression.

    Args:
        source (str): String expression to parse.

    Returns:
        Tuple[Callable, Tuple[str]]: Returns a callable function and tuple of string symbols.

    Raises:
        QiskitError: If expression is not parsable.
    """
    subs = [('numpy.', ''), ('np.', ''), ('math.', '')]
    for match, sub in subs:
        source = source.replace(match, sub)

    params = sorted(re.findall(_param_regex, source))
    try:
        expr = ast.parse(source, mode='eval')
    except SyntaxError:
        raise QiskitError('%s is invalid expression.' % source)

    def evaluated_func(*args, **kwargs):
        locals_dict = {}

        if args:
            for key, val in zip(params, args):
                locals_dict[key] = val
        if kwargs:
            locals_dict.update({key: val for key, val in
                                kwargs.items() if key in params})

        if sorted(locals_dict.keys()) != params:
            raise QiskitError('Supplied params ({args}, {kwargs}) do not match '
                              '{params}'.format(args=args, kwargs=kwargs, params=params))

        return _eval_expr(expr, locals_dict)

    return evaluated_func, params
