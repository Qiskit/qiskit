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

from .ast import (
    ASTNode,
    AliasStatement,
    BitDeclaration,
    BranchingStatement,
    CalibrationDefinition,
    CalibrationGrammarDeclaration,
    ComparisonExpression,
    Constant,
    Designator,
    EqualsOperator,
    Expression,
    GtOperator,
    Header,
    IO,
    IOModifier,
    Identifier,
    Include,
    Integer,
    LtOperator,
    PhysicalQubitIdentifier,
    Pragma,
    Program,
    ProgramBlock,
    QuantumArgument,
    QuantumBarrier,
    QuantumDeclaration,
    QuantumGateCall,
    QuantumGateDefinition,
    QuantumGateModifier,
    QuantumGateModifierName,
    QuantumGateSignature,
    QuantumMeasurement,
    QuantumMeasurementAssignment,
    QuantumReset,
    Range,
    ReturnStatement,
    SubroutineCall,
    SubroutineDefinition,
    SubscriptedIdentifier,
    Version,
)


class BasicPrinter:
    """A QASM 3 AST visitor which writes the tree out in text mode to a stream, where the only
    formatting is simple block indentation."""

    _CONSTANT_LOOKUP = {
        Constant.pi: "pi",
        Constant.euler: "euler",
        Constant.tau: "tau",
    }

    _MODIFIER_LOOKUP = {
        QuantumGateModifierName.ctrl: "ctrl",
        QuantumGateModifierName.negctrl: "negctrl",
        QuantumGateModifierName.inv: "inv",
        QuantumGateModifierName.pow: "pow",
    }

    # The visitor names include the class names, so they mix snake_case with PascalCase.
    # pylint: disable=invalid-name

    def __init__(self, stream: io.TextIOBase, *, indent: str):
        """
        Args:
            stream (io.TextIOBase): the stream that the output will be written to.
            indent (str): the string to use as a single indentation level.
        """
        self.stream = stream
        self.indent = indent
        self._current_indent = 0

    def visit(self, node: ASTNode) -> None:
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
        self, nodes: Sequence[ASTNode], *, start: str = "", end: str = "", separator: str
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

    def _visit_Program(self, node: Program) -> None:
        self.visit(node.header)
        for statement in node.statements:
            self.visit(statement)

    def _visit_Header(self, node: Header) -> None:
        self.visit(node.version)
        for include in node.includes:
            self.visit(include)

    def _visit_Version(self, node: Version) -> None:
        self._write_statement(f"OPENQASM {node.version_number}")

    def _visit_Include(self, node: Include) -> None:
        self._write_statement(f'include "{node.filename}"')

    def _visit_Pragma(self, node: Pragma) -> None:
        self._write_statement(f"#pragma {node.content}")

    def _visit_CalibrationGrammarDeclaration(self, node: CalibrationGrammarDeclaration) -> None:
        self._write_statement(f'defcalgrammar "{node.name}"')

    def _visit_Identifier(self, node: Identifier) -> None:
        self.stream.write(node.string)

    def _visit_PhysicalQubitIdentifier(self, node: PhysicalQubitIdentifier) -> None:
        self.stream.write("$")
        self.visit(node.identifier)

    def _visit_Expression(self, node: Expression) -> None:
        self.stream.write(str(node.something))

    def _visit_Constant(self, node: Constant) -> None:
        self.stream.write(self._CONSTANT_LOOKUP[node])

    def _visit_SubscriptedIdentifier(self, node: SubscriptedIdentifier) -> None:
        self.visit(node.identifier)
        self.stream.write("[")
        self.visit(node.subscript)
        self.stream.write("]")

    def _visit_Range(self, node: Range) -> None:
        if node.start is not None:
            self.visit(node.start)
        self.stream.write(":")
        if node.step is not None:
            self.visit(node.step)
            self.stream.write(":")
        if node.end is not None:
            self.visit(node.end)

    def _visit_QuantumMeasurement(self, node: QuantumMeasurement) -> None:
        self.stream.write("measure ")
        self._visit_sequence(node.identifierList, separator=", ")

    def _visit_QuantumMeasurementAssignment(self, node: QuantumMeasurementAssignment) -> None:
        self._start_line()
        self.visit(node.identifier)
        self.stream.write(" = ")
        self.visit(node.quantumMeasurement)
        self._end_statement()

    def _visit_QuantumReset(self, node: QuantumReset) -> None:
        self._start_line()
        self.stream.write("reset ")
        self.visit(node.identifier)
        self._end_statement()

    def _visit_Integer(self, node: Integer) -> None:
        self.stream.write(str(node.something))

    def _visit_Designator(self, node: Designator) -> None:
        self.stream.write("[")
        self.visit(node.expression)
        self.stream.write("]")

    def _visit_BitDeclaration(self, node: BitDeclaration) -> None:
        self._start_line()
        self.stream.write("bit")
        self.visit(node.designator)
        self.stream.write(" ")
        self.visit(node.identifier)
        if node.equalsExpression:
            self.stream.write(" ")
            self.visit(node.equalsExpression)
        self._end_statement()

    def _visit_QuantumDeclaration(self, node: QuantumDeclaration) -> None:
        self._start_line()
        self.stream.write("qubit")
        self.visit(node.designator)
        self.stream.write(" ")
        self.visit(node.identifier)
        self._end_statement()

    def _visit_AliasStatement(self, node: AliasStatement) -> None:
        self._start_line()
        self.stream.write("let ")
        self.visit(node.identifier)
        self.stream.write(" = ")
        self._visit_sequence(node.concatenation, separator=" ++ ")
        self._end_statement()

    def _visit_QuantumGateModifier(self, node: QuantumGateModifier) -> None:
        self.stream.write(self._MODIFIER_LOOKUP[node.modifier])
        if node.argument:
            self.stream.write("(")
            self.visit(node.argument)
            self.stream.write(")")

    def _visit_QuantumGateCall(self, node: QuantumGateCall) -> None:
        self._start_line()
        if node.modifiers:
            self._visit_sequence(node.modifiers, end=" @ ", separator=" @ ")
        self.visit(node.quantumGateName)
        if node.parameters:
            self._visit_sequence(node.parameters, start="(", end=")", separator=", ")
        self.stream.write(" ")
        self._visit_sequence(node.indexIdentifierList, separator=", ")
        self._end_statement()

    def _visit_SubroutineCall(self, node: SubroutineCall) -> None:
        self._start_line()
        self.visit(node.identifier)
        if node.expressionList:
            self._visit_sequence(node.expressionList, start="(", end=")", separator=", ")
        self.stream.write(" ")
        self._visit_sequence(node.indexIdentifierList, separator=", ")
        self._end_statement()

    def _visit_QuantumBarrier(self, node: QuantumBarrier) -> None:
        self._start_line()
        self.stream.write("barrier ")
        self._visit_sequence(node.indexIdentifierList, separator=", ")
        self._end_statement()

    def _visit_ProgramBlock(self, node: ProgramBlock) -> None:
        self.stream.write("{\n")
        self._current_indent += 1
        for statement in node.statements:
            self.visit(statement)
        self._current_indent -= 1
        self._start_line()
        self.stream.write("}")

    def _visit_ReturnStatement(self, node: ReturnStatement) -> None:
        self._start_line()
        if node.expression:
            self.stream.write("return ")
            self.visit(node.expression)
        else:
            self.stream.write("return")
        self._end_statement()

    def _visit_QuantumArgument(self, node: QuantumArgument) -> None:
        self.stream.write("qubit")
        if node.designator:
            self.visit(node.designator)
        self.stream.write(" ")
        self.visit(node.identifier)

    def _visit_QuantumGateSignature(self, node: QuantumGateSignature) -> None:
        self.visit(node.name)
        if node.params:
            self._visit_sequence(node.params, start="(", end=")", separator=", ")
        self.stream.write(" ")
        self._visit_sequence(node.qargList, separator=", ")

    def _visit_QuantumGateDefinition(self, node: QuantumGateDefinition) -> None:
        self._start_line()
        self.stream.write("gate ")
        self.visit(node.quantumGateSignature)
        self.stream.write(" ")
        self.visit(node.quantumBlock)
        self._end_line()

    def _visit_SubroutineDefinition(self, node: SubroutineDefinition) -> None:
        self._start_line()
        self.stream.write("def ")
        self.visit(node.identifier)
        self._visit_sequence(node.arguments, start="(", end=")", separator=", ")
        self.stream.write(" ")
        self.visit(node.subroutineBlock)
        self._end_line()

    def _visit_CalibrationDefinition(self, node: CalibrationDefinition) -> None:
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

    def _visit_LtOperator(self, _node: LtOperator) -> None:
        self.stream.write(">")

    def _visit_GtOperator(self, _node: GtOperator) -> None:
        self.stream.write("<")

    def _visit_EqualsOperator(self, _node: EqualsOperator) -> None:
        self.stream.write("==")

    def _visit_ComparisonExpression(self, node: ComparisonExpression) -> None:
        self.visit(node.left)
        self.stream.write(" ")
        self.visit(node.relation)
        self.stream.write(" ")
        self.visit(node.right)

    def _visit_BranchingStatement(self, node: BranchingStatement) -> None:
        self._start_line()
        self.stream.write("if (")
        self.visit(node.booleanExpression)
        self.stream.write(") ")
        self.visit(node.programTrue)
        if node.programFalse:
            self.stream.write(" else ")
            self.visit(node.programFalse)
        self._end_line()

    def _visit_IO(self, node: IO) -> None:
        self._start_line()
        modifier = "input" if node.modifier == IOModifier.input else "output"
        self.stream.write(modifier + " ")
        self.visit(node.type)
        self.stream.write(" ")
        self.visit(node.variable)
        self._end_statement()
