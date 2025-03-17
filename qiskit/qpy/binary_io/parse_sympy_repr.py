# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Parser for sympy expressions srepr from ParameterExpression internals."""

import ast

from qiskit.qpy.exceptions import QpyError


ALLOWED_CALLERS = {
    "Abs",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Symbol",
    "Integer",
    "Rational",
    "Complex",
    "Float",
    "log",
    "sin",
    "cos",
    "tan",
    "atan",
    "acos",
    "asin",
    "exp",
    "conjugate",
}

UNARY = {
    "sin",
    "cos",
    "tan",
    "atan",
    "acos",
    "asin",
    "conjugate",
    "Symbol",
    "Integer",
    "Complex",
    "Abs",
    "Float",
}


class ParseSympyWalker(ast.NodeVisitor):
    """A custom ast walker that is passed the sympy srepr from QPY < 13 and creates a custom
    expression."""

    def __init__(self):
        self.stack = []

    def visit_UnaryOp(self, node: ast.UnaryOp):  # pylint: disable=invalid-name
        """Visit a python unary op node"""
        self.visit(node.operand)
        arg = self.stack.pop()
        if isinstance(node.op, ast.UAdd):
            self.stack.append(+arg)
        elif isinstance(node.op, ast.USub):
            self.stack.append(-arg)
        elif isinstance(node.op, ast.Not):
            self.stack.append(not arg)
        elif isinstance(node.op, ast.Invert):
            self.stack.append(~arg)
        else:
            raise QpyError(f"Invalid unary op as part of sympy srepr: {node.op}")

    def visit_Constant(self, node: ast.Constant):  # pylint: disable=invalid-name
        """Visit a constant node."""
        self.stack.append(node.value)

    def visit_Call(self, node: ast.Call):  # pylint: disable=invalid-name
        """Visit a call node

        This can only be parameter expression allowed sympy call types.
        """
        import sympy

        if isinstance(node.func, ast.Name):
            name = node.func.id
        else:
            raise QpyError("Unknown node type")

        if name not in ALLOWED_CALLERS:
            raise QpyError(f"{name} is not part of a valid sympy expression srepr")

        args = node.args
        if name in UNARY:
            if len(args) != 1:
                raise QpyError(f"{name} has an invalid number of args in sympy srepr")
            self.visit(args[0])
            obj = getattr(sympy, name)(self.stack.pop())
            self.stack.append(obj)
        else:
            for arg in args:
                self.visit(arg)
            out_args = [self.stack.pop() for _ in range(len(args))]
            out_args.reverse()
            obj = getattr(sympy, name)(*out_args)
            self.stack.append(obj)


def parse_sympy_repr(sympy_repr: str):
    """Parse a given sympy srepr into a symbolic expression object."""
    tree = ast.parse(sympy_repr, mode="eval")
    visitor = ParseSympyWalker()
    visitor.visit(tree)
    return visitor.stack.pop()
