# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Node visitor as defined in https://docs.python.org/3/library/ast.html#ast.NodeVisitor
This module is used internally by ``qiskit.transpiler.classicalfunction.ClassicalFunction``.
"""

import ast
import _ast

from tweedledum.classical import LogicNetwork

from .exceptions import ClassicalFunctionParseError, ClassicalFunctionCompilerTypeError


class ClassicalFunctionVisitor(ast.NodeVisitor):
    """Node visitor as defined in https://docs.python.org/3/library/ast.html#ast.NodeVisitor"""

    # pylint: disable=invalid-name
    bitops = {
        _ast.BitAnd: "create_and",
        _ast.BitOr: "create_or",
        _ast.BitXor: "create_xor",
        _ast.And: "create_and",
        _ast.Or: "create_or",
        _ast.Not: "create_not",
    }

    def __init__(self):
        self.scopes = []
        self.args = []
        self._network = None
        self.name = None
        super().__init__()

    def visit_Module(self, node):
        """The full snippet should contain a single function"""
        if len(node.body) != 1 and not isinstance(node.body[0], ast.FunctionDef):
            raise ClassicalFunctionParseError("just functions, sorry!")
        self.name = node.body[0].name
        self.visit(node.body[0])

    def visit_FunctionDef(self, node):
        """The function definition should have type hints"""
        if node.returns is None:
            raise ClassicalFunctionParseError("return type is needed")
        scope = {"return": (node.returns.id, None), node.returns.id: ("type", None)}

        # Extend scope with the decorator's names
        scope.update({decorator.id: ("decorator", None) for decorator in node.decorator_list})

        self.scopes.append(scope)
        self._network = LogicNetwork()
        self.extend_scope(node.args)
        return super().generic_visit(node)

    def visit_Return(self, node):
        """The return type should match the return type hint."""
        _type, signal = self.visit(node.value)
        if _type != self.scopes[-1]["return"][0]:
            raise ClassicalFunctionParseError("return type error")
        self._network.create_po(signal)

    def visit_Assign(self, node):
        """When assign, the scope needs to be updated with the right type"""
        type_value, signal_value = self.visit(node.value)
        for target in node.targets:
            self.scopes[-1][target.id] = (type_value, signal_value)
        return (type_value, signal_value)

    def bit_binop(self, op, values):
        """Uses ClassicalFunctionVisitor.bitops to extend self._network"""
        bitop = ClassicalFunctionVisitor.bitops.get(type(op))
        if not bitop:
            raise ClassicalFunctionParseError("Unknown binop.op %s" % op)
        binop = getattr(self._network, bitop)

        left_type, left_signal = values[0]
        if left_type != "Int1":
            raise ClassicalFunctionParseError("binop type error")

        for right_type, right_signal in values[1:]:
            if right_type != "Int1":
                raise ClassicalFunctionParseError("binop type error")
            left_signal = binop(left_signal, right_signal)

        return "Int1", left_signal

    def visit_BoolOp(self, node):
        """Handles ``and`` and ``or``.
        node.left=Int1 and node.right=Int1 return Int1"""
        return self.bit_binop(node.op, [self.visit(value) for value in node.values])

    def visit_BinOp(self, node):
        """Handles ``&``, ``^``, and ``|``.
        node.left=Int1 and node.right=Int1 return Int1"""
        return self.bit_binop(node.op, [self.visit(node.left), self.visit(node.right)])

    def visit_UnaryOp(self, node):
        """Handles ``~``. Cannot operate on Int1s."""
        operand_type, operand_signal = self.visit(node.operand)
        if operand_type != "Int1":
            raise ClassicalFunctionCompilerTypeError(
                "UntaryOp.op %s only support operation on Int1s for now" % node.op
            )
        bitop = ClassicalFunctionVisitor.bitops.get(type(node.op))
        if not bitop:
            raise ClassicalFunctionCompilerTypeError(
                "UntaryOp.op %s does not operate with Int1 type " % node.op
            )
        return "Int1", getattr(self._network, bitop)(operand_signal)

    def visit_Name(self, node):
        """Reduce variable names."""
        if node.id not in self.scopes[-1]:
            raise ClassicalFunctionParseError("out of scope: %s" % node.id)
        return self.scopes[-1][node.id]

    def generic_visit(self, node):
        """Catch all for the unhandled nodes."""
        if isinstance(
            node,
            (
                _ast.arguments,
                _ast.arg,
                _ast.Load,
                _ast.BitAnd,
                _ast.BitOr,
                _ast.BitXor,
                _ast.BoolOp,
                _ast.Or,
            ),
        ):
            return super().generic_visit(node)
        raise ClassicalFunctionParseError("Unknown node: %s" % type(node))

    def extend_scope(self, args_node: _ast.arguments) -> None:
        """Add the arguments to the scope"""
        for arg in args_node.args:
            if arg.annotation is None:
                raise ClassicalFunctionParseError("argument type is needed")
            self.args.append(arg.arg)
            self.scopes[-1][arg.annotation.id] = ("type", None)
            self.scopes[-1][arg.arg] = (arg.annotation.id, self._network.create_pi())
