# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Printers for QASM 3 AST nodes."""

import io
from typing import Sequence

from . import ast


class BasicPrinter:
    """A QASM 3 AST visitor which writes the tree out in text mode to a stream, where the only
    formatting is simple block indentation."""

    _CONSTANT_LOOKUP = {
        ast.Constant.PI: "pi",
        ast.Constant.EULER: "euler",
        ast.Constant.TAU: "tau",
    }

    _MODIFIER_LOOKUP = {
        ast.QuantumGateModifierName.CTRL: "ctrl",
        ast.QuantumGateModifierName.NEGCTRL: "negctrl",
        ast.QuantumGateModifierName.INV: "inv",
        ast.QuantumGateModifierName.POW: "pow",
    }

    _FLOAT_WIDTH_LOOKUP = {type: str(type.value) for type in ast.FloatType}

    # The visitor names include the class names, so they mix snake_case with PascalCase.
    # pylint: disable=invalid-name

    def __init__(self, stream: io.TextIOBase, *, indent: str, chain_else_if: bool = False):
        """
        Args:
            stream (io.TextIOBase): the stream that the output will be written to.
            indent (str): the string to use as a single indentation level.
            chain_else_if (bool): If ``True``, then constructs of the form::

                    if (x == 0) {
                        // ...
                    } else {
                        if (x == 1) {
                            // ...
                        } else {
                            // ...
                        }
                    }

                will be collapsed into the equivalent but flatter::

                    if (x == 0) {
                        // ...
                    } else if (x == 1) {
                        // ...
                    } else {
                        // ...
                    }

                This collapsed form may have less support on backends, so it is turned off by
                default.  While the output of this printer is always unambiguous, using ``else``
                without immediately opening an explicit scope with ``{ }`` in nested contexts can
                cause issues, in the general case, which is why it is sometimes less supported.
        """
        self.stream = stream
        self.indent = indent
        self._current_indent = 0
        self._chain_else_if = chain_else_if

    def visit(self, node: ast.ASTNode) -> None:
        """Visit this node of the AST, printing it out to the stream in this class instance.

        Normally, you will want to call this function on a complete :obj:`~qiskit.qasm3.ast.Program`
        node, to print out a complete program to the stream.  The visit can start from any node,
        however, if you want to build up a file bit-by-bit manually.

        Args:
            node (ASTNode): the node to convert to QASM 3 and write out to the stream.

        Raises:
            RuntimeError: if an AST node is encountered that the visitor is unable to parse.  This
                typically means that the given AST was malformed.
        """
        visitor = None
        for cls_ in type(node).mro():
            visitor = getattr(self, "_visit_" + cls_.__name__, None)
            if visitor is not None:
                break
        else:
            node_type = node.__class__.__name__
            raise RuntimeError(
                f"This visitor does not know how to handle AST nodes of type '{node_type}',"
                f" but was given '{node}'."
            )
        visitor(node)

    def _start_line(self) -> None:
        self.stream.write(self._current_indent * self.indent)

    def _end_statement(self) -> None:
        self.stream.write(";\n")

    def _end_line(self) -> None:
        self.stream.write("\n")

    def _write_statement(self, line: str) -> None:
        self._start_line()
        self.stream.write(line)
        self._end_statement()

    def _visit_sequence(
        self, nodes: Sequence[ast.ASTNode], *, start: str = "", end: str = "", separator: str
    ) -> None:
        if start:
            self.stream.write(start)
        for node in nodes[:-1]:
            self.visit(node)
            self.stream.write(separator)
        if nodes:
            self.visit(nodes[-1])
        if end:
            self.stream.write(end)

    def _visit_Program(self, node: ast.Program) -> None:
        self.visit(node.header)
        for statement in node.statements:
            self.visit(statement)

    def _visit_Header(self, node: ast.Header) -> None:
        self.visit(node.version)
        for include in node.includes:
            self.visit(include)

    def _visit_Version(self, node: ast.Version) -> None:
        self._write_statement(f"OPENQASM {node.version_number}")

    def _visit_Include(self, node: ast.Include) -> None:
        self._write_statement(f'include "{node.filename}"')

    def _visit_Pragma(self, node: ast.Pragma) -> None:
        self._write_statement(f"#pragma {node.content}")

    def _visit_CalibrationGrammarDeclaration(self, node: ast.CalibrationGrammarDeclaration) -> None:
        self._write_statement(f'defcalgrammar "{node.name}"')

    def _visit_FloatType(self, node: ast.FloatType) -> None:
        self.stream.write(f"float[{self._FLOAT_WIDTH_LOOKUP[node]}]")

    def _visit_BitArrayType(self, node: ast.BitArrayType) -> None:
        self.stream.write(f"bit[{node.size}]")

    def _visit_Identifier(self, node: ast.Identifier) -> None:
        self.stream.write(node.string)

    def _visit_PhysicalQubitIdentifier(self, node: ast.PhysicalQubitIdentifier) -> None:
        self.stream.write("$")
        self.visit(node.identifier)

    def _visit_Expression(self, node: ast.Expression) -> None:
        self.stream.write(str(node.something))

    def _visit_Constant(self, node: ast.Constant) -> None:
        self.stream.write(self._CONSTANT_LOOKUP[node])

    def _visit_SubscriptedIdentifier(self, node: ast.SubscriptedIdentifier) -> None:
        self.visit(node.identifier)
        self.stream.write("[")
        self.visit(node.subscript)
        self.stream.write("]")

    def _visit_Range(self, node: ast.Range) -> None:
        if node.start is not None:
            self.visit(node.start)
        self.stream.write(":")
        if node.step is not None:
            self.visit(node.step)
            self.stream.write(":")
        if node.end is not None:
            self.visit(node.end)

    def _visit_IndexSet(self, node: ast.IndexSet) -> None:
        self._visit_sequence(node.values, start="{", separator=", ", end="}")

    def _visit_QuantumMeasurement(self, node: ast.QuantumMeasurement) -> None:
        self.stream.write("measure ")
        self._visit_sequence(node.identifierList, separator=", ")

    def _visit_QuantumMeasurementAssignment(self, node: ast.QuantumMeasurementAssignment) -> None:
        self._start_line()
        self.visit(node.identifier)
        self.stream.write(" = ")
        self.visit(node.quantumMeasurement)
        self._end_statement()

    def _visit_QuantumReset(self, node: ast.QuantumReset) -> None:
        self._start_line()
        self.stream.write("reset ")
        self.visit(node.identifier)
        self._end_statement()

    def _visit_Integer(self, node: ast.Integer) -> None:
        self.stream.write(str(node.something))

    def _visit_Designator(self, node: ast.Designator) -> None:
        self.stream.write("[")
        self.visit(node.expression)
        self.stream.write("]")

    def _visit_ClassicalDeclaration(self, node: ast.ClassicalDeclaration) -> None:
        self._start_line()
        self.visit(node.type)
        self.stream.write(" ")
        self.visit(node.identifier)
        if node.initializer is not None:
            self.stream.write(" = ")
            self.visit(node.initializer)
        self._end_statement()

    def _visit_IODeclaration(self, node: ast.IODeclaration) -> None:
        self._start_line()
        modifier = "input" if node.modifier is ast.IOModifier.INPUT else "output"
        self.stream.write(modifier + " ")
        self.visit(node.type)
        self.stream.write(" ")
        self.visit(node.identifier)
        self._end_statement()

    def _visit_QuantumDeclaration(self, node: ast.QuantumDeclaration) -> None:
        self._start_line()
        self.stream.write("qubit")
        self.visit(node.designator)
        self.stream.write(" ")
        self.visit(node.identifier)
        self._end_statement()

    def _visit_AliasStatement(self, node: ast.AliasStatement) -> None:
        self._start_line()
        self.stream.write("let ")
        self.visit(node.identifier)
        self.stream.write(" = ")
        self._visit_sequence(node.concatenation, separator=" ++ ")
        self._end_statement()

    def _visit_QuantumGateModifier(self, node: ast.QuantumGateModifier) -> None:
        self.stream.write(self._MODIFIER_LOOKUP[node.modifier])
        if node.argument:
            self.stream.write("(")
            self.visit(node.argument)
            self.stream.write(")")

    def _visit_QuantumGateCall(self, node: ast.QuantumGateCall) -> None:
        self._start_line()
        if node.modifiers:
            self._visit_sequence(node.modifiers, end=" @ ", separator=" @ ")
        self.visit(node.quantumGateName)
        if node.parameters:
            self._visit_sequence(node.parameters, start="(", end=")", separator=", ")
        self.stream.write(" ")
        self._visit_sequence(node.indexIdentifierList, separator=", ")
        self._end_statement()

    def _visit_SubroutineCall(self, node: ast.SubroutineCall) -> None:
        self._start_line()
        self.visit(node.identifier)
        if node.expressionList:
            self._visit_sequence(node.expressionList, start="(", end=")", separator=", ")
        self.stream.write(" ")
        self._visit_sequence(node.indexIdentifierList, separator=", ")
        self._end_statement()

    def _visit_QuantumBarrier(self, node: ast.QuantumBarrier) -> None:
        self._start_line()
        self.stream.write("barrier ")
        self._visit_sequence(node.indexIdentifierList, separator=", ")
        self._end_statement()

    def _visit_ProgramBlock(self, node: ast.ProgramBlock) -> None:
        self.stream.write("{\n")
        self._current_indent += 1
        for statement in node.statements:
            self.visit(statement)
        self._current_indent -= 1
        self._start_line()
        self.stream.write("}")

    def _visit_ReturnStatement(self, node: ast.ReturnStatement) -> None:
        self._start_line()
        if node.expression:
            self.stream.write("return ")
            self.visit(node.expression)
        else:
            self.stream.write("return")
        self._end_statement()

    def _visit_QuantumArgument(self, node: ast.QuantumArgument) -> None:
        self.stream.write("qubit")
        if node.designator:
            self.visit(node.designator)
        self.stream.write(" ")
        self.visit(node.identifier)

    def _visit_QuantumGateSignature(self, node: ast.QuantumGateSignature) -> None:
        self.visit(node.name)
        if node.params:
            self._visit_sequence(node.params, start="(", end=")", separator=", ")
        self.stream.write(" ")
        self._visit_sequence(node.qargList, separator=", ")

    def _visit_QuantumGateDefinition(self, node: ast.QuantumGateDefinition) -> None:
        self._start_line()
        self.stream.write("gate ")
        self.visit(node.quantumGateSignature)
        self.stream.write(" ")
        self.visit(node.quantumBlock)
        self._end_line()

    def _visit_SubroutineDefinition(self, node: ast.SubroutineDefinition) -> None:
        self._start_line()
        self.stream.write("def ")
        self.visit(node.identifier)
        self._visit_sequence(node.arguments, start="(", end=")", separator=", ")
        self.stream.write(" ")
        self.visit(node.subroutineBlock)
        self._end_line()

    def _visit_CalibrationDefinition(self, node: ast.CalibrationDefinition) -> None:
        self._start_line()
        self.stream.write("defcal ")
        self.visit(node.name)
        self.stream.write(" ")
        if node.calibrationArgumentList:
            self._visit_sequence(node.calibrationArgumentList, start="(", end=")", separator=", ")
            self.stream.write(" ")
        self._visit_sequence(node.identifierList, separator=", ")
        # This is temporary: calibration definition blocks are not currently (2021-10-04) defined
        # properly.
        self.stream.write(" {}")
        self._end_line()

    def _visit_LtOperator(self, _node: ast.LtOperator) -> None:
        self.stream.write(">")

    def _visit_GtOperator(self, _node: ast.GtOperator) -> None:
        self.stream.write("<")

    def _visit_EqualsOperator(self, _node: ast.EqualsOperator) -> None:
        self.stream.write("==")

    def _visit_ComparisonExpression(self, node: ast.ComparisonExpression) -> None:
        self.visit(node.left)
        self.stream.write(" ")
        self.visit(node.relation)
        self.stream.write(" ")
        self.visit(node.right)

    def _visit_BreakStatement(self, _node: ast.BreakStatement) -> None:
        self._write_statement("break")

    def _visit_ContinueStatement(self, _node: ast.ContinueStatement) -> None:
        self._write_statement("continue")

    def _visit_BranchingStatement(
        self, node: ast.BranchingStatement, chained: bool = False
    ) -> None:
        if not chained:
            self._start_line()
        self.stream.write("if (")
        self.visit(node.condition)
        self.stream.write(") ")
        self.visit(node.true_body)
        if node.false_body is not None:
            self.stream.write(" else ")
            # Special handling to flatten a perfectly nested structure of
            #   if {...} else { if {...} else {...} }
            # into the simpler
            #   if {...} else if {...} else {...}
            # but only if we're allowed to by our options.
            if (
                self._chain_else_if
                and len(node.false_body.statements) == 1
                and isinstance(node.false_body.statements[0], ast.BranchingStatement)
            ):
                self._visit_BranchingStatement(node.false_body.statements[0], chained=True)
            else:
                self.visit(node.false_body)
        if not chained:
            # The visitor to the first ``if`` will end the line.
            self._end_line()

    def _visit_ForLoopStatement(self, node: ast.ForLoopStatement) -> None:
        self._start_line()
        self.stream.write("for ")
        self.visit(node.parameter)
        self.stream.write(" in ")
        if isinstance(node.indexset, ast.Range):
            self.stream.write("[")
            self.visit(node.indexset)
            self.stream.write("]")
        else:
            self.visit(node.indexset)
        self.stream.write(" ")
        self.visit(node.body)
        self._end_line()

    def _visit_WhileLoopStatement(self, node: ast.WhileLoopStatement) -> None:
        self._start_line()
        self.stream.write("while (")
        self.visit(node.condition)
        self.stream.write(") ")
        self.visit(node.body)
        self._end_line()
