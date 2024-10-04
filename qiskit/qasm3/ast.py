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

# pylint: disable=invalid-name, super-init-not-called, missing-class-docstring, redefined-builtin

"""QASM3 AST Nodes"""

import enum
from typing import Optional, List, Union, Iterable, Tuple


class ASTNode:
    """Base abstract class for AST nodes"""


class Statement(ASTNode):
    """
    statement
        : expressionStatement
        | assignmentStatement
        | classicalDeclarationStatement
        | branchingStatement
        | loopStatement
        | endStatement
        | aliasStatement
        | quantumStatement
    """


class Pragma(ASTNode):
    """
    pragma
        : '#pragma' LBRACE statement* RBRACE  // match any valid openqasm statement
    """

    def __init__(self, content):
        self.content = content


class CalibrationGrammarDeclaration(Statement):
    """
    calibrationGrammarDeclaration
        : 'defcalgrammar' calibrationGrammar SEMICOLON
    """

    def __init__(self, name):
        self.name = name


class Program(ASTNode):
    """
    program
        : header (globalStatement | statement)*
    """

    def __init__(self, header, statements=None):
        self.header = header
        self.statements = statements or []


class Header(ASTNode):
    """
    header
        : version? include*
    """

    def __init__(self, version, includes):
        self.version = version
        self.includes = includes


class Include(ASTNode):
    """
    include
        : 'include' StringLiteral SEMICOLON
    """

    def __init__(self, filename):
        self.filename = filename


class Version(ASTNode):
    """
    version
        : 'OPENQASM'(Integer | RealNumber) SEMICOLON
    """

    def __init__(self, version_number):
        self.version_number = version_number


class QuantumInstruction(ASTNode):
    """
    quantumInstruction
        : quantumGateCall
        | quantumPhase
        | quantumMeasurement
        | quantumReset
        | quantumBarrier
    """


class ClassicalType(ASTNode):
    """Information about a classical type.  This is just an abstract base for inheritance tests."""


class FloatType(ClassicalType, enum.Enum):
    """Allowed values for the width of floating-point types."""

    HALF = 16
    SINGLE = 32
    DOUBLE = 64
    QUAD = 128
    OCT = 256


class BoolType(ClassicalType):
    """Type information for a Boolean."""


class IntType(ClassicalType):
    """Type information for a signed integer."""

    def __init__(self, size: Optional[int] = None):
        self.size = size


class UintType(ClassicalType):
    """Type information for an unsigned integer."""

    def __init__(self, size: Optional[int] = None):
        self.size = size


class BitType(ClassicalType):
    """Type information for a single bit."""


class BitArrayType(ClassicalType):
    """Type information for a sized number of classical bits."""

    def __init__(self, size: int):
        self.size = size


class Expression(ASTNode):
    pass


class StringifyAndPray(Expression):
    # This is not a real AST node, yet is somehow very common. It's used when there are
    # `ParameterExpression` instances; instead of actually visiting the Sympy expression tree into
    # an OQ3 AST, we just convert it to a string, cross our fingers, and hope.
    def __init__(self, obj):
        self.obj = obj


class Range(Expression):
    def __init__(
        self,
        start: Optional[Expression] = None,
        end: Optional[Expression] = None,
        step: Optional[Expression] = None,
    ):
        self.start = start
        self.step = step
        self.end = end


class Identifier(Expression):
    def __init__(self, string):
        self.string = string


class SubscriptedIdentifier(Identifier):
    """An identifier with subscripted access."""

    def __init__(self, string: str, subscript: Union[Range, Expression]):
        super().__init__(string)
        self.subscript = subscript


class Constant(Expression, enum.Enum):
    """A constant value defined by the QASM 3 spec."""

    PI = enum.auto()
    EULER = enum.auto()
    TAU = enum.auto()


class IntegerLiteral(Expression):
    def __init__(self, value):
        self.value = value


class BooleanLiteral(Expression):
    def __init__(self, value):
        self.value = value


class BitstringLiteral(Expression):
    def __init__(self, value, width):
        self.value = value
        self.width = width


class DurationUnit(enum.Enum):
    """Valid values for the unit of durations."""

    NANOSECOND = "ns"
    MICROSECOND = "us"
    MILLISECOND = "ms"
    SECOND = "s"
    SAMPLE = "dt"


class DurationLiteral(Expression):
    def __init__(self, value: float, unit: DurationUnit):
        self.value = value
        self.unit = unit


class Unary(Expression):
    class Op(enum.Enum):
        LOGIC_NOT = "!"
        BIT_NOT = "~"

    def __init__(self, op: Op, operand: Expression):
        self.op = op
        self.operand = operand


class Binary(Expression):
    class Op(enum.Enum):
        BIT_AND = "&"
        BIT_OR = "|"
        BIT_XOR = "^"
        LOGIC_AND = "&&"
        LOGIC_OR = "||"
        LESS = "<"
        LESS_EQUAL = "<="
        GREATER = ">"
        GREATER_EQUAL = ">="
        EQUAL = "=="
        NOT_EQUAL = "!="
        SHIFT_LEFT = "<<"
        SHIFT_RIGHT = ">>"

    def __init__(self, op: Op, left: Expression, right: Expression):
        self.op = op
        self.left = left
        self.right = right


class Cast(Expression):
    def __init__(self, type: ClassicalType, operand: Expression):
        self.type = type
        self.operand = operand


class Index(Expression):
    def __init__(self, target: Expression, index: Expression):
        self.target = target
        self.index = index


class IndexSet(ASTNode):
    """
    A literal index set of values::

        { Expression (, Expression)* }
    """

    def __init__(self, values: List[Expression]):
        self.values = values


class QuantumMeasurement(ASTNode):
    """
    quantumMeasurement
        : 'measure' indexIdentifierList
    """

    def __init__(self, identifierList: List[Identifier]):
        self.identifierList = identifierList


class QuantumMeasurementAssignment(Statement):
    """
    quantumMeasurementAssignment
        : quantumMeasurement ARROW indexIdentifierList
        | indexIdentifier EQUALS quantumMeasurement  # eg: bits = measure qubits;
    """

    def __init__(self, identifier: Identifier, quantumMeasurement: QuantumMeasurement):
        self.identifier = identifier
        self.quantumMeasurement = quantumMeasurement


class Designator(ASTNode):
    """
    designator
        : LBRACKET expression RBRACKET
    """

    def __init__(self, expression: Expression):
        self.expression = expression


class ClassicalDeclaration(Statement):
    """Declaration of a classical type, optionally initializing it to a value."""

    def __init__(self, type_: ClassicalType, identifier: Identifier, initializer=None):
        self.type = type_
        self.identifier = identifier
        self.initializer = initializer


class AssignmentStatement(Statement):
    """Assignment of an expression to an l-value."""

    def __init__(self, lvalue: SubscriptedIdentifier, rvalue: Expression):
        self.lvalue = lvalue
        self.rvalue = rvalue


class QuantumDeclaration(ASTNode):
    """
    quantumDeclaration
        : 'qreg' Identifier designator? |   # NOT SUPPORTED
         'qubit' designator? Identifier
    """

    def __init__(self, identifier: Identifier, designator=None):
        self.identifier = identifier
        self.designator = designator


class AliasStatement(ASTNode):
    """
    aliasStatement
        : 'let' Identifier EQUALS indexIdentifier SEMICOLON
    """

    def __init__(self, identifier: Identifier, value: Expression):
        self.identifier = identifier
        self.value = value


class QuantumGateModifierName(enum.Enum):
    """The names of the allowed modifiers of quantum gates."""

    CTRL = enum.auto()
    NEGCTRL = enum.auto()
    INV = enum.auto()
    POW = enum.auto()


class QuantumGateModifier(ASTNode):
    """A modifier of a gate. For example, in ``ctrl @ x $0``, the ``ctrl @`` is the modifier."""

    def __init__(self, modifier: QuantumGateModifierName, argument: Optional[Expression] = None):
        self.modifier = modifier
        self.argument = argument


class QuantumGateCall(QuantumInstruction):
    """
    quantumGateCall
        : quantumGateModifier* quantumGateName ( LPAREN expressionList? RPAREN )? indexIdentifierList
    """

    def __init__(
        self,
        quantumGateName: Identifier,
        indexIdentifierList: List[Identifier],
        parameters: List[Expression] = None,
        modifiers: Optional[List[QuantumGateModifier]] = None,
    ):
        self.quantumGateName = quantumGateName
        self.indexIdentifierList = indexIdentifierList
        self.parameters = parameters or []
        self.modifiers = modifiers or []


class QuantumBarrier(QuantumInstruction):
    """
    quantumBarrier
        : 'barrier' indexIdentifierList
    """

    def __init__(self, indexIdentifierList: List[Identifier]):
        self.indexIdentifierList = indexIdentifierList


class QuantumReset(QuantumInstruction):
    """A built-in ``reset q0;`` statement."""

    def __init__(self, identifier: Identifier):
        self.identifier = identifier


class QuantumDelay(QuantumInstruction):
    """A built-in ``delay[duration] q0;`` statement."""

    def __init__(self, duration: Expression, qubits: List[Identifier]):
        self.duration = duration
        self.qubits = qubits


class ProgramBlock(ASTNode):
    """
    programBlock
        : statement | controlDirective
        | LBRACE(statement | controlDirective) * RBRACE
    """

    def __init__(self, statements: List[Statement]):
        self.statements = statements


class ReturnStatement(ASTNode):  # TODO probably should be a subclass of ControlDirective
    """
    returnStatement
        : 'return' ( expression | quantumMeasurement )? SEMICOLON;
    """

    def __init__(self, expression=None):
        self.expression = expression


class QuantumBlock(ProgramBlock):
    """
    quantumBlock
        : LBRACE ( quantumStatement | quantumLoop )* RBRACE
    """

    pass


class SubroutineBlock(ProgramBlock):
    """
    subroutineBlock
        : LBRACE statement* returnStatement? RBRACE
    """

    pass


class QuantumGateDefinition(Statement):
    """
    quantumGateDefinition
        : 'gate' quantumGateSignature quantumBlock
    """

    def __init__(
        self,
        name: Identifier,
        params: Tuple[Identifier, ...],
        qubits: Tuple[Identifier, ...],
        body: QuantumBlock,
    ):
        self.name = name
        self.params = params
        self.qubits = qubits
        self.body = body


class SubroutineDefinition(Statement):
    """
    subroutineDefinition
        : 'def' Identifier LPAREN anyTypeArgumentList? RPAREN
        returnSignature? subroutineBlock
    """

    def __init__(
        self,
        identifier: Identifier,
        subroutineBlock: SubroutineBlock,
        arguments=None,  # [ClassicalArgument]
    ):
        self.identifier = identifier
        self.arguments = arguments or []
        self.subroutineBlock = subroutineBlock


class CalibrationArgument(ASTNode):
    """
    calibrationArgumentList
        : classicalArgumentList | expressionList
    """

    pass


class CalibrationDefinition(Statement):
    """
    calibrationDefinition
        : 'defcal' Identifier
        ( LPAREN calibrationArgumentList? RPAREN )? identifierList
        returnSignature? LBRACE .*? RBRACE  // for now, match anything inside body
        ;
    """

    def __init__(
        self,
        name: Identifier,
        identifierList: List[Identifier],
        calibrationArgumentList: Optional[List[CalibrationArgument]] = None,
    ):
        self.name = name
        self.identifierList = identifierList
        self.calibrationArgumentList = calibrationArgumentList or []


class BranchingStatement(Statement):
    """
    branchingStatement
        : 'if' LPAREN booleanExpression RPAREN programBlock ( 'else' programBlock )?
    """

    def __init__(self, condition: Expression, true_body: ProgramBlock, false_body=None):
        self.condition = condition
        self.true_body = true_body
        self.false_body = false_body


class ForLoopStatement(Statement):
    """
    AST node for ``for`` loops.

    ::

        ForLoop: "for" Identifier "in" SetDeclaration ProgramBlock
        SetDeclaration:
            | Identifier
            | "{" Expression ("," Expression)* "}"
            | "[" Range "]"
    """

    def __init__(
        self,
        indexset: Union[Identifier, IndexSet, Range],
        parameter: Identifier,
        body: ProgramBlock,
    ):
        self.indexset = indexset
        self.parameter = parameter
        self.body = body


class WhileLoopStatement(Statement):
    """
    AST node for ``while`` loops.

    ::

        WhileLoop: "while" "(" Expression ")" ProgramBlock
    """

    def __init__(self, condition: Expression, body: ProgramBlock):
        self.condition = condition
        self.body = body


class BreakStatement(Statement):
    """AST node for ``break`` statements.  Has no associated information."""


class ContinueStatement(Statement):
    """AST node for ``continue`` statements.  Has no associated information."""


class IOModifier(enum.Enum):
    """IO Modifier object"""

    INPUT = enum.auto()
    OUTPUT = enum.auto()


class IODeclaration(ClassicalDeclaration):
    """A declaration of an IO variable."""

    def __init__(self, modifier: IOModifier, type_: ClassicalType, identifier: Identifier):
        super().__init__(type_, identifier)
        self.modifier = modifier


class DefaultCase(Expression):
    """An object representing the `default` special label in switch statements."""


class SwitchStatementPreview(Statement):
    """AST node for the proposed 'switch-case' extension to OpenQASM 3, before the syntax was
    stabilized.  This corresponds to the :attr:`.ExperimentalFeatures.SWITCH_CASE_V1` logic.

    The stabilized form of the syntax instead uses :class:`.SwitchStatement`."""

    def __init__(
        self, target: Expression, cases: Iterable[Tuple[Iterable[Expression], ProgramBlock]]
    ):
        self.target = target
        self.cases = [(tuple(values), case) for values, case in cases]


class SwitchStatement(Statement):
    """AST node for the stable 'switch' statement of OpenQASM 3.

    The only real difference from an AST form is that the default is required to be separate; it
    cannot be joined with other cases (even though that's meaningless, the V1 syntax permitted it).
    """

    def __init__(
        self,
        target: Expression,
        cases: Iterable[Tuple[Iterable[Expression], ProgramBlock]],
        default: Optional[ProgramBlock] = None,
    ):
        self.target = target
        self.cases = [(tuple(values), case) for values, case in cases]
        self.default = default
