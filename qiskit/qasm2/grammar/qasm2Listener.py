# Generated from qasm2.g4 by ANTLR 4.9.2
from antlr4 import *

if __name__ is not None and "." in __name__:
    from .qasm2Parser import qasm2Parser
else:
    from qasm2Parser import qasm2Parser

# This class defines a complete listener for a parse tree produced by qasm2Parser.
class qasm2Listener(ParseTreeListener):

    # Enter a parse tree produced by qasm2Parser#program.
    def enterProgram(self, ctx: qasm2Parser.ProgramContext):
        pass

    # Exit a parse tree produced by qasm2Parser#program.
    def exitProgram(self, ctx: qasm2Parser.ProgramContext):
        pass

    # Enter a parse tree produced by qasm2Parser#header.
    def enterHeader(self, ctx: qasm2Parser.HeaderContext):
        pass

    # Exit a parse tree produced by qasm2Parser#header.
    def exitHeader(self, ctx: qasm2Parser.HeaderContext):
        pass

    # Enter a parse tree produced by qasm2Parser#version.
    def enterVersion(self, ctx: qasm2Parser.VersionContext):
        pass

    # Exit a parse tree produced by qasm2Parser#version.
    def exitVersion(self, ctx: qasm2Parser.VersionContext):
        pass

    # Enter a parse tree produced by qasm2Parser#include.
    def enterInclude(self, ctx: qasm2Parser.IncludeContext):
        pass

    # Exit a parse tree produced by qasm2Parser#include.
    def exitInclude(self, ctx: qasm2Parser.IncludeContext):
        pass

    # Enter a parse tree produced by qasm2Parser#ioIdentifier.
    def enterIoIdentifier(self, ctx: qasm2Parser.IoIdentifierContext):
        pass

    # Exit a parse tree produced by qasm2Parser#ioIdentifier.
    def exitIoIdentifier(self, ctx: qasm2Parser.IoIdentifierContext):
        pass

    # Enter a parse tree produced by qasm2Parser#io.
    def enterIo(self, ctx: qasm2Parser.IoContext):
        pass

    # Exit a parse tree produced by qasm2Parser#io.
    def exitIo(self, ctx: qasm2Parser.IoContext):
        pass

    # Enter a parse tree produced by qasm2Parser#globalStatement.
    def enterGlobalStatement(self, ctx: qasm2Parser.GlobalStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#globalStatement.
    def exitGlobalStatement(self, ctx: qasm2Parser.GlobalStatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#statement.
    def enterStatement(self, ctx: qasm2Parser.StatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#statement.
    def exitStatement(self, ctx: qasm2Parser.StatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#quantumDeclarationStatement.
    def enterQuantumDeclarationStatement(
        self, ctx: qasm2Parser.QuantumDeclarationStatementContext
    ):
        pass

    # Exit a parse tree produced by qasm2Parser#quantumDeclarationStatement.
    def exitQuantumDeclarationStatement(
        self, ctx: qasm2Parser.QuantumDeclarationStatementContext
    ):
        pass

    # Enter a parse tree produced by qasm2Parser#classicalDeclarationStatement.
    def enterClassicalDeclarationStatement(
        self, ctx: qasm2Parser.ClassicalDeclarationStatementContext
    ):
        pass

    # Exit a parse tree produced by qasm2Parser#classicalDeclarationStatement.
    def exitClassicalDeclarationStatement(
        self, ctx: qasm2Parser.ClassicalDeclarationStatementContext
    ):
        pass

    # Enter a parse tree produced by qasm2Parser#classicalAssignment.
    def enterClassicalAssignment(self, ctx: qasm2Parser.ClassicalAssignmentContext):
        pass

    # Exit a parse tree produced by qasm2Parser#classicalAssignment.
    def exitClassicalAssignment(self, ctx: qasm2Parser.ClassicalAssignmentContext):
        pass

    # Enter a parse tree produced by qasm2Parser#assignmentStatement.
    def enterAssignmentStatement(self, ctx: qasm2Parser.AssignmentStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#assignmentStatement.
    def exitAssignmentStatement(self, ctx: qasm2Parser.AssignmentStatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#returnSignature.
    def enterReturnSignature(self, ctx: qasm2Parser.ReturnSignatureContext):
        pass

    # Exit a parse tree produced by qasm2Parser#returnSignature.
    def exitReturnSignature(self, ctx: qasm2Parser.ReturnSignatureContext):
        pass

    # Enter a parse tree produced by qasm2Parser#designator.
    def enterDesignator(self, ctx: qasm2Parser.DesignatorContext):
        pass

    # Exit a parse tree produced by qasm2Parser#designator.
    def exitDesignator(self, ctx: qasm2Parser.DesignatorContext):
        pass

    # Enter a parse tree produced by qasm2Parser#doubleDesignator.
    def enterDoubleDesignator(self, ctx: qasm2Parser.DoubleDesignatorContext):
        pass

    # Exit a parse tree produced by qasm2Parser#doubleDesignator.
    def exitDoubleDesignator(self, ctx: qasm2Parser.DoubleDesignatorContext):
        pass

    # Enter a parse tree produced by qasm2Parser#identifierList.
    def enterIdentifierList(self, ctx: qasm2Parser.IdentifierListContext):
        pass

    # Exit a parse tree produced by qasm2Parser#identifierList.
    def exitIdentifierList(self, ctx: qasm2Parser.IdentifierListContext):
        pass

    # Enter a parse tree produced by qasm2Parser#quantumDeclaration.
    def enterQuantumDeclaration(self, ctx: qasm2Parser.QuantumDeclarationContext):
        pass

    # Exit a parse tree produced by qasm2Parser#quantumDeclaration.
    def exitQuantumDeclaration(self, ctx: qasm2Parser.QuantumDeclarationContext):
        pass

    # Enter a parse tree produced by qasm2Parser#quantumArgument.
    def enterQuantumArgument(self, ctx: qasm2Parser.QuantumArgumentContext):
        pass

    # Exit a parse tree produced by qasm2Parser#quantumArgument.
    def exitQuantumArgument(self, ctx: qasm2Parser.QuantumArgumentContext):
        pass

    # Enter a parse tree produced by qasm2Parser#quantumArgumentList.
    def enterQuantumArgumentList(self, ctx: qasm2Parser.QuantumArgumentListContext):
        pass

    # Exit a parse tree produced by qasm2Parser#quantumArgumentList.
    def exitQuantumArgumentList(self, ctx: qasm2Parser.QuantumArgumentListContext):
        pass

    # Enter a parse tree produced by qasm2Parser#bitType.
    def enterBitType(self, ctx: qasm2Parser.BitTypeContext):
        pass

    # Exit a parse tree produced by qasm2Parser#bitType.
    def exitBitType(self, ctx: qasm2Parser.BitTypeContext):
        pass

    # Enter a parse tree produced by qasm2Parser#singleDesignatorType.
    def enterSingleDesignatorType(self, ctx: qasm2Parser.SingleDesignatorTypeContext):
        pass

    # Exit a parse tree produced by qasm2Parser#singleDesignatorType.
    def exitSingleDesignatorType(self, ctx: qasm2Parser.SingleDesignatorTypeContext):
        pass

    # Enter a parse tree produced by qasm2Parser#doubleDesignatorType.
    def enterDoubleDesignatorType(self, ctx: qasm2Parser.DoubleDesignatorTypeContext):
        pass

    # Exit a parse tree produced by qasm2Parser#doubleDesignatorType.
    def exitDoubleDesignatorType(self, ctx: qasm2Parser.DoubleDesignatorTypeContext):
        pass

    # Enter a parse tree produced by qasm2Parser#noDesignatorType.
    def enterNoDesignatorType(self, ctx: qasm2Parser.NoDesignatorTypeContext):
        pass

    # Exit a parse tree produced by qasm2Parser#noDesignatorType.
    def exitNoDesignatorType(self, ctx: qasm2Parser.NoDesignatorTypeContext):
        pass

    # Enter a parse tree produced by qasm2Parser#classicalType.
    def enterClassicalType(self, ctx: qasm2Parser.ClassicalTypeContext):
        pass

    # Exit a parse tree produced by qasm2Parser#classicalType.
    def exitClassicalType(self, ctx: qasm2Parser.ClassicalTypeContext):
        pass

    # Enter a parse tree produced by qasm2Parser#constantDeclaration.
    def enterConstantDeclaration(self, ctx: qasm2Parser.ConstantDeclarationContext):
        pass

    # Exit a parse tree produced by qasm2Parser#constantDeclaration.
    def exitConstantDeclaration(self, ctx: qasm2Parser.ConstantDeclarationContext):
        pass

    # Enter a parse tree produced by qasm2Parser#singleDesignatorDeclaration.
    def enterSingleDesignatorDeclaration(
        self, ctx: qasm2Parser.SingleDesignatorDeclarationContext
    ):
        pass

    # Exit a parse tree produced by qasm2Parser#singleDesignatorDeclaration.
    def exitSingleDesignatorDeclaration(
        self, ctx: qasm2Parser.SingleDesignatorDeclarationContext
    ):
        pass

    # Enter a parse tree produced by qasm2Parser#doubleDesignatorDeclaration.
    def enterDoubleDesignatorDeclaration(
        self, ctx: qasm2Parser.DoubleDesignatorDeclarationContext
    ):
        pass

    # Exit a parse tree produced by qasm2Parser#doubleDesignatorDeclaration.
    def exitDoubleDesignatorDeclaration(
        self, ctx: qasm2Parser.DoubleDesignatorDeclarationContext
    ):
        pass

    # Enter a parse tree produced by qasm2Parser#noDesignatorDeclaration.
    def enterNoDesignatorDeclaration(
        self, ctx: qasm2Parser.NoDesignatorDeclarationContext
    ):
        pass

    # Exit a parse tree produced by qasm2Parser#noDesignatorDeclaration.
    def exitNoDesignatorDeclaration(
        self, ctx: qasm2Parser.NoDesignatorDeclarationContext
    ):
        pass

    # Enter a parse tree produced by qasm2Parser#bitDeclaration.
    def enterBitDeclaration(self, ctx: qasm2Parser.BitDeclarationContext):
        pass

    # Exit a parse tree produced by qasm2Parser#bitDeclaration.
    def exitBitDeclaration(self, ctx: qasm2Parser.BitDeclarationContext):
        pass

    # Enter a parse tree produced by qasm2Parser#classicalDeclaration.
    def enterClassicalDeclaration(self, ctx: qasm2Parser.ClassicalDeclarationContext):
        pass

    # Exit a parse tree produced by qasm2Parser#classicalDeclaration.
    def exitClassicalDeclaration(self, ctx: qasm2Parser.ClassicalDeclarationContext):
        pass

    # Enter a parse tree produced by qasm2Parser#classicalTypeList.
    def enterClassicalTypeList(self, ctx: qasm2Parser.ClassicalTypeListContext):
        pass

    # Exit a parse tree produced by qasm2Parser#classicalTypeList.
    def exitClassicalTypeList(self, ctx: qasm2Parser.ClassicalTypeListContext):
        pass

    # Enter a parse tree produced by qasm2Parser#classicalArgument.
    def enterClassicalArgument(self, ctx: qasm2Parser.ClassicalArgumentContext):
        pass

    # Exit a parse tree produced by qasm2Parser#classicalArgument.
    def exitClassicalArgument(self, ctx: qasm2Parser.ClassicalArgumentContext):
        pass

    # Enter a parse tree produced by qasm2Parser#classicalArgumentList.
    def enterClassicalArgumentList(self, ctx: qasm2Parser.ClassicalArgumentListContext):
        pass

    # Exit a parse tree produced by qasm2Parser#classicalArgumentList.
    def exitClassicalArgumentList(self, ctx: qasm2Parser.ClassicalArgumentListContext):
        pass

    # Enter a parse tree produced by qasm2Parser#aliasStatement.
    def enterAliasStatement(self, ctx: qasm2Parser.AliasStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#aliasStatement.
    def exitAliasStatement(self, ctx: qasm2Parser.AliasStatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#indexIdentifier.
    def enterIndexIdentifier(self, ctx: qasm2Parser.IndexIdentifierContext):
        pass

    # Exit a parse tree produced by qasm2Parser#indexIdentifier.
    def exitIndexIdentifier(self, ctx: qasm2Parser.IndexIdentifierContext):
        pass

    # Enter a parse tree produced by qasm2Parser#indexIdentifierList.
    def enterIndexIdentifierList(self, ctx: qasm2Parser.IndexIdentifierListContext):
        pass

    # Exit a parse tree produced by qasm2Parser#indexIdentifierList.
    def exitIndexIdentifierList(self, ctx: qasm2Parser.IndexIdentifierListContext):
        pass

    # Enter a parse tree produced by qasm2Parser#rangeDefinition.
    def enterRangeDefinition(self, ctx: qasm2Parser.RangeDefinitionContext):
        pass

    # Exit a parse tree produced by qasm2Parser#rangeDefinition.
    def exitRangeDefinition(self, ctx: qasm2Parser.RangeDefinitionContext):
        pass

    # Enter a parse tree produced by qasm2Parser#quantumGateDefinition.
    def enterQuantumGateDefinition(self, ctx: qasm2Parser.QuantumGateDefinitionContext):
        pass

    # Exit a parse tree produced by qasm2Parser#quantumGateDefinition.
    def exitQuantumGateDefinition(self, ctx: qasm2Parser.QuantumGateDefinitionContext):
        pass

    # Enter a parse tree produced by qasm2Parser#quantumGateSignature.
    def enterQuantumGateSignature(self, ctx: qasm2Parser.QuantumGateSignatureContext):
        pass

    # Exit a parse tree produced by qasm2Parser#quantumGateSignature.
    def exitQuantumGateSignature(self, ctx: qasm2Parser.QuantumGateSignatureContext):
        pass

    # Enter a parse tree produced by qasm2Parser#quantumGateName.
    def enterQuantumGateName(self, ctx: qasm2Parser.QuantumGateNameContext):
        pass

    # Exit a parse tree produced by qasm2Parser#quantumGateName.
    def exitQuantumGateName(self, ctx: qasm2Parser.QuantumGateNameContext):
        pass

    # Enter a parse tree produced by qasm2Parser#quantumBlock.
    def enterQuantumBlock(self, ctx: qasm2Parser.QuantumBlockContext):
        pass

    # Exit a parse tree produced by qasm2Parser#quantumBlock.
    def exitQuantumBlock(self, ctx: qasm2Parser.QuantumBlockContext):
        pass

    # Enter a parse tree produced by qasm2Parser#quantumLoop.
    def enterQuantumLoop(self, ctx: qasm2Parser.QuantumLoopContext):
        pass

    # Exit a parse tree produced by qasm2Parser#quantumLoop.
    def exitQuantumLoop(self, ctx: qasm2Parser.QuantumLoopContext):
        pass

    # Enter a parse tree produced by qasm2Parser#quantumLoopBlock.
    def enterQuantumLoopBlock(self, ctx: qasm2Parser.QuantumLoopBlockContext):
        pass

    # Exit a parse tree produced by qasm2Parser#quantumLoopBlock.
    def exitQuantumLoopBlock(self, ctx: qasm2Parser.QuantumLoopBlockContext):
        pass

    # Enter a parse tree produced by qasm2Parser#quantumStatement.
    def enterQuantumStatement(self, ctx: qasm2Parser.QuantumStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#quantumStatement.
    def exitQuantumStatement(self, ctx: qasm2Parser.QuantumStatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#quantumInstruction.
    def enterQuantumInstruction(self, ctx: qasm2Parser.QuantumInstructionContext):
        pass

    # Exit a parse tree produced by qasm2Parser#quantumInstruction.
    def exitQuantumInstruction(self, ctx: qasm2Parser.QuantumInstructionContext):
        pass

    # Enter a parse tree produced by qasm2Parser#quantumPhase.
    def enterQuantumPhase(self, ctx: qasm2Parser.QuantumPhaseContext):
        pass

    # Exit a parse tree produced by qasm2Parser#quantumPhase.
    def exitQuantumPhase(self, ctx: qasm2Parser.QuantumPhaseContext):
        pass

    # Enter a parse tree produced by qasm2Parser#quantumReset.
    def enterQuantumReset(self, ctx: qasm2Parser.QuantumResetContext):
        pass

    # Exit a parse tree produced by qasm2Parser#quantumReset.
    def exitQuantumReset(self, ctx: qasm2Parser.QuantumResetContext):
        pass

    # Enter a parse tree produced by qasm2Parser#quantumMeasurement.
    def enterQuantumMeasurement(self, ctx: qasm2Parser.QuantumMeasurementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#quantumMeasurement.
    def exitQuantumMeasurement(self, ctx: qasm2Parser.QuantumMeasurementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#quantumMeasurementAssignment.
    def enterQuantumMeasurementAssignment(
        self, ctx: qasm2Parser.QuantumMeasurementAssignmentContext
    ):
        pass

    # Exit a parse tree produced by qasm2Parser#quantumMeasurementAssignment.
    def exitQuantumMeasurementAssignment(
        self, ctx: qasm2Parser.QuantumMeasurementAssignmentContext
    ):
        pass

    # Enter a parse tree produced by qasm2Parser#quantumBarrier.
    def enterQuantumBarrier(self, ctx: qasm2Parser.QuantumBarrierContext):
        pass

    # Exit a parse tree produced by qasm2Parser#quantumBarrier.
    def exitQuantumBarrier(self, ctx: qasm2Parser.QuantumBarrierContext):
        pass

    # Enter a parse tree produced by qasm2Parser#quantumGateModifier.
    def enterQuantumGateModifier(self, ctx: qasm2Parser.QuantumGateModifierContext):
        pass

    # Exit a parse tree produced by qasm2Parser#quantumGateModifier.
    def exitQuantumGateModifier(self, ctx: qasm2Parser.QuantumGateModifierContext):
        pass

    # Enter a parse tree produced by qasm2Parser#powModifier.
    def enterPowModifier(self, ctx: qasm2Parser.PowModifierContext):
        pass

    # Exit a parse tree produced by qasm2Parser#powModifier.
    def exitPowModifier(self, ctx: qasm2Parser.PowModifierContext):
        pass

    # Enter a parse tree produced by qasm2Parser#ctrlModifier.
    def enterCtrlModifier(self, ctx: qasm2Parser.CtrlModifierContext):
        pass

    # Exit a parse tree produced by qasm2Parser#ctrlModifier.
    def exitCtrlModifier(self, ctx: qasm2Parser.CtrlModifierContext):
        pass

    # Enter a parse tree produced by qasm2Parser#quantumGateCall.
    def enterQuantumGateCall(self, ctx: qasm2Parser.QuantumGateCallContext):
        pass

    # Exit a parse tree produced by qasm2Parser#quantumGateCall.
    def exitQuantumGateCall(self, ctx: qasm2Parser.QuantumGateCallContext):
        pass

    # Enter a parse tree produced by qasm2Parser#unaryOperator.
    def enterUnaryOperator(self, ctx: qasm2Parser.UnaryOperatorContext):
        pass

    # Exit a parse tree produced by qasm2Parser#unaryOperator.
    def exitUnaryOperator(self, ctx: qasm2Parser.UnaryOperatorContext):
        pass

    # Enter a parse tree produced by qasm2Parser#comparisonOperator.
    def enterComparisonOperator(self, ctx: qasm2Parser.ComparisonOperatorContext):
        pass

    # Exit a parse tree produced by qasm2Parser#comparisonOperator.
    def exitComparisonOperator(self, ctx: qasm2Parser.ComparisonOperatorContext):
        pass

    # Enter a parse tree produced by qasm2Parser#equalityOperator.
    def enterEqualityOperator(self, ctx: qasm2Parser.EqualityOperatorContext):
        pass

    # Exit a parse tree produced by qasm2Parser#equalityOperator.
    def exitEqualityOperator(self, ctx: qasm2Parser.EqualityOperatorContext):
        pass

    # Enter a parse tree produced by qasm2Parser#logicalOperator.
    def enterLogicalOperator(self, ctx: qasm2Parser.LogicalOperatorContext):
        pass

    # Exit a parse tree produced by qasm2Parser#logicalOperator.
    def exitLogicalOperator(self, ctx: qasm2Parser.LogicalOperatorContext):
        pass

    # Enter a parse tree produced by qasm2Parser#expressionStatement.
    def enterExpressionStatement(self, ctx: qasm2Parser.ExpressionStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#expressionStatement.
    def exitExpressionStatement(self, ctx: qasm2Parser.ExpressionStatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#expression.
    def enterExpression(self, ctx: qasm2Parser.ExpressionContext):
        pass

    # Exit a parse tree produced by qasm2Parser#expression.
    def exitExpression(self, ctx: qasm2Parser.ExpressionContext):
        pass

    # Enter a parse tree produced by qasm2Parser#logicalAndExpression.
    def enterLogicalAndExpression(self, ctx: qasm2Parser.LogicalAndExpressionContext):
        pass

    # Exit a parse tree produced by qasm2Parser#logicalAndExpression.
    def exitLogicalAndExpression(self, ctx: qasm2Parser.LogicalAndExpressionContext):
        pass

    # Enter a parse tree produced by qasm2Parser#bitOrExpression.
    def enterBitOrExpression(self, ctx: qasm2Parser.BitOrExpressionContext):
        pass

    # Exit a parse tree produced by qasm2Parser#bitOrExpression.
    def exitBitOrExpression(self, ctx: qasm2Parser.BitOrExpressionContext):
        pass

    # Enter a parse tree produced by qasm2Parser#xOrExpression.
    def enterXOrExpression(self, ctx: qasm2Parser.XOrExpressionContext):
        pass

    # Exit a parse tree produced by qasm2Parser#xOrExpression.
    def exitXOrExpression(self, ctx: qasm2Parser.XOrExpressionContext):
        pass

    # Enter a parse tree produced by qasm2Parser#bitAndExpression.
    def enterBitAndExpression(self, ctx: qasm2Parser.BitAndExpressionContext):
        pass

    # Exit a parse tree produced by qasm2Parser#bitAndExpression.
    def exitBitAndExpression(self, ctx: qasm2Parser.BitAndExpressionContext):
        pass

    # Enter a parse tree produced by qasm2Parser#equalityExpression.
    def enterEqualityExpression(self, ctx: qasm2Parser.EqualityExpressionContext):
        pass

    # Exit a parse tree produced by qasm2Parser#equalityExpression.
    def exitEqualityExpression(self, ctx: qasm2Parser.EqualityExpressionContext):
        pass

    # Enter a parse tree produced by qasm2Parser#comparisonExpression.
    def enterComparisonExpression(self, ctx: qasm2Parser.ComparisonExpressionContext):
        pass

    # Exit a parse tree produced by qasm2Parser#comparisonExpression.
    def exitComparisonExpression(self, ctx: qasm2Parser.ComparisonExpressionContext):
        pass

    # Enter a parse tree produced by qasm2Parser#bitShiftExpression.
    def enterBitShiftExpression(self, ctx: qasm2Parser.BitShiftExpressionContext):
        pass

    # Exit a parse tree produced by qasm2Parser#bitShiftExpression.
    def exitBitShiftExpression(self, ctx: qasm2Parser.BitShiftExpressionContext):
        pass

    # Enter a parse tree produced by qasm2Parser#additiveExpression.
    def enterAdditiveExpression(self, ctx: qasm2Parser.AdditiveExpressionContext):
        pass

    # Exit a parse tree produced by qasm2Parser#additiveExpression.
    def exitAdditiveExpression(self, ctx: qasm2Parser.AdditiveExpressionContext):
        pass

    # Enter a parse tree produced by qasm2Parser#multiplicativeExpression.
    def enterMultiplicativeExpression(
        self, ctx: qasm2Parser.MultiplicativeExpressionContext
    ):
        pass

    # Exit a parse tree produced by qasm2Parser#multiplicativeExpression.
    def exitMultiplicativeExpression(
        self, ctx: qasm2Parser.MultiplicativeExpressionContext
    ):
        pass

    # Enter a parse tree produced by qasm2Parser#unaryExpression.
    def enterUnaryExpression(self, ctx: qasm2Parser.UnaryExpressionContext):
        pass

    # Exit a parse tree produced by qasm2Parser#unaryExpression.
    def exitUnaryExpression(self, ctx: qasm2Parser.UnaryExpressionContext):
        pass

    # Enter a parse tree produced by qasm2Parser#powerExpression.
    def enterPowerExpression(self, ctx: qasm2Parser.PowerExpressionContext):
        pass

    # Exit a parse tree produced by qasm2Parser#powerExpression.
    def exitPowerExpression(self, ctx: qasm2Parser.PowerExpressionContext):
        pass

    # Enter a parse tree produced by qasm2Parser#expressionTerminator.
    def enterExpressionTerminator(self, ctx: qasm2Parser.ExpressionTerminatorContext):
        pass

    # Exit a parse tree produced by qasm2Parser#expressionTerminator.
    def exitExpressionTerminator(self, ctx: qasm2Parser.ExpressionTerminatorContext):
        pass

    # Enter a parse tree produced by qasm2Parser#booleanLiteral.
    def enterBooleanLiteral(self, ctx: qasm2Parser.BooleanLiteralContext):
        pass

    # Exit a parse tree produced by qasm2Parser#booleanLiteral.
    def exitBooleanLiteral(self, ctx: qasm2Parser.BooleanLiteralContext):
        pass

    # Enter a parse tree produced by qasm2Parser#incrementor.
    def enterIncrementor(self, ctx: qasm2Parser.IncrementorContext):
        pass

    # Exit a parse tree produced by qasm2Parser#incrementor.
    def exitIncrementor(self, ctx: qasm2Parser.IncrementorContext):
        pass

    # Enter a parse tree produced by qasm2Parser#builtInCall.
    def enterBuiltInCall(self, ctx: qasm2Parser.BuiltInCallContext):
        pass

    # Exit a parse tree produced by qasm2Parser#builtInCall.
    def exitBuiltInCall(self, ctx: qasm2Parser.BuiltInCallContext):
        pass

    # Enter a parse tree produced by qasm2Parser#builtInMath.
    def enterBuiltInMath(self, ctx: qasm2Parser.BuiltInMathContext):
        pass

    # Exit a parse tree produced by qasm2Parser#builtInMath.
    def exitBuiltInMath(self, ctx: qasm2Parser.BuiltInMathContext):
        pass

    # Enter a parse tree produced by qasm2Parser#castOperator.
    def enterCastOperator(self, ctx: qasm2Parser.CastOperatorContext):
        pass

    # Exit a parse tree produced by qasm2Parser#castOperator.
    def exitCastOperator(self, ctx: qasm2Parser.CastOperatorContext):
        pass

    # Enter a parse tree produced by qasm2Parser#expressionList.
    def enterExpressionList(self, ctx: qasm2Parser.ExpressionListContext):
        pass

    # Exit a parse tree produced by qasm2Parser#expressionList.
    def exitExpressionList(self, ctx: qasm2Parser.ExpressionListContext):
        pass

    # Enter a parse tree produced by qasm2Parser#equalsExpression.
    def enterEqualsExpression(self, ctx: qasm2Parser.EqualsExpressionContext):
        pass

    # Exit a parse tree produced by qasm2Parser#equalsExpression.
    def exitEqualsExpression(self, ctx: qasm2Parser.EqualsExpressionContext):
        pass

    # Enter a parse tree produced by qasm2Parser#assignmentOperator.
    def enterAssignmentOperator(self, ctx: qasm2Parser.AssignmentOperatorContext):
        pass

    # Exit a parse tree produced by qasm2Parser#assignmentOperator.
    def exitAssignmentOperator(self, ctx: qasm2Parser.AssignmentOperatorContext):
        pass

    # Enter a parse tree produced by qasm2Parser#setDeclaration.
    def enterSetDeclaration(self, ctx: qasm2Parser.SetDeclarationContext):
        pass

    # Exit a parse tree produced by qasm2Parser#setDeclaration.
    def exitSetDeclaration(self, ctx: qasm2Parser.SetDeclarationContext):
        pass

    # Enter a parse tree produced by qasm2Parser#programBlock.
    def enterProgramBlock(self, ctx: qasm2Parser.ProgramBlockContext):
        pass

    # Exit a parse tree produced by qasm2Parser#programBlock.
    def exitProgramBlock(self, ctx: qasm2Parser.ProgramBlockContext):
        pass

    # Enter a parse tree produced by qasm2Parser#branchingStatement.
    def enterBranchingStatement(self, ctx: qasm2Parser.BranchingStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#branchingStatement.
    def exitBranchingStatement(self, ctx: qasm2Parser.BranchingStatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#loopSignature.
    def enterLoopSignature(self, ctx: qasm2Parser.LoopSignatureContext):
        pass

    # Exit a parse tree produced by qasm2Parser#loopSignature.
    def exitLoopSignature(self, ctx: qasm2Parser.LoopSignatureContext):
        pass

    # Enter a parse tree produced by qasm2Parser#loopStatement.
    def enterLoopStatement(self, ctx: qasm2Parser.LoopStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#loopStatement.
    def exitLoopStatement(self, ctx: qasm2Parser.LoopStatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#endStatement.
    def enterEndStatement(self, ctx: qasm2Parser.EndStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#endStatement.
    def exitEndStatement(self, ctx: qasm2Parser.EndStatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#returnStatement.
    def enterReturnStatement(self, ctx: qasm2Parser.ReturnStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#returnStatement.
    def exitReturnStatement(self, ctx: qasm2Parser.ReturnStatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#controlDirective.
    def enterControlDirective(self, ctx: qasm2Parser.ControlDirectiveContext):
        pass

    # Exit a parse tree produced by qasm2Parser#controlDirective.
    def exitControlDirective(self, ctx: qasm2Parser.ControlDirectiveContext):
        pass

    # Enter a parse tree produced by qasm2Parser#externDeclaration.
    def enterExternDeclaration(self, ctx: qasm2Parser.ExternDeclarationContext):
        pass

    # Exit a parse tree produced by qasm2Parser#externDeclaration.
    def exitExternDeclaration(self, ctx: qasm2Parser.ExternDeclarationContext):
        pass

    # Enter a parse tree produced by qasm2Parser#externCall.
    def enterExternCall(self, ctx: qasm2Parser.ExternCallContext):
        pass

    # Exit a parse tree produced by qasm2Parser#externCall.
    def exitExternCall(self, ctx: qasm2Parser.ExternCallContext):
        pass

    # Enter a parse tree produced by qasm2Parser#subroutineDefinition.
    def enterSubroutineDefinition(self, ctx: qasm2Parser.SubroutineDefinitionContext):
        pass

    # Exit a parse tree produced by qasm2Parser#subroutineDefinition.
    def exitSubroutineDefinition(self, ctx: qasm2Parser.SubroutineDefinitionContext):
        pass

    # Enter a parse tree produced by qasm2Parser#subroutineBlock.
    def enterSubroutineBlock(self, ctx: qasm2Parser.SubroutineBlockContext):
        pass

    # Exit a parse tree produced by qasm2Parser#subroutineBlock.
    def exitSubroutineBlock(self, ctx: qasm2Parser.SubroutineBlockContext):
        pass

    # Enter a parse tree produced by qasm2Parser#subroutineCall.
    def enterSubroutineCall(self, ctx: qasm2Parser.SubroutineCallContext):
        pass

    # Exit a parse tree produced by qasm2Parser#subroutineCall.
    def exitSubroutineCall(self, ctx: qasm2Parser.SubroutineCallContext):
        pass

    # Enter a parse tree produced by qasm2Parser#pragma.
    def enterPragma(self, ctx: qasm2Parser.PragmaContext):
        pass

    # Exit a parse tree produced by qasm2Parser#pragma.
    def exitPragma(self, ctx: qasm2Parser.PragmaContext):
        pass

    # Enter a parse tree produced by qasm2Parser#timingType.
    def enterTimingType(self, ctx: qasm2Parser.TimingTypeContext):
        pass

    # Exit a parse tree produced by qasm2Parser#timingType.
    def exitTimingType(self, ctx: qasm2Parser.TimingTypeContext):
        pass

    # Enter a parse tree produced by qasm2Parser#timingBox.
    def enterTimingBox(self, ctx: qasm2Parser.TimingBoxContext):
        pass

    # Exit a parse tree produced by qasm2Parser#timingBox.
    def exitTimingBox(self, ctx: qasm2Parser.TimingBoxContext):
        pass

    # Enter a parse tree produced by qasm2Parser#timingIdentifier.
    def enterTimingIdentifier(self, ctx: qasm2Parser.TimingIdentifierContext):
        pass

    # Exit a parse tree produced by qasm2Parser#timingIdentifier.
    def exitTimingIdentifier(self, ctx: qasm2Parser.TimingIdentifierContext):
        pass

    # Enter a parse tree produced by qasm2Parser#timingInstructionName.
    def enterTimingInstructionName(self, ctx: qasm2Parser.TimingInstructionNameContext):
        pass

    # Exit a parse tree produced by qasm2Parser#timingInstructionName.
    def exitTimingInstructionName(self, ctx: qasm2Parser.TimingInstructionNameContext):
        pass

    # Enter a parse tree produced by qasm2Parser#timingInstruction.
    def enterTimingInstruction(self, ctx: qasm2Parser.TimingInstructionContext):
        pass

    # Exit a parse tree produced by qasm2Parser#timingInstruction.
    def exitTimingInstruction(self, ctx: qasm2Parser.TimingInstructionContext):
        pass

    # Enter a parse tree produced by qasm2Parser#timingStatement.
    def enterTimingStatement(self, ctx: qasm2Parser.TimingStatementContext):
        pass

    # Exit a parse tree produced by qasm2Parser#timingStatement.
    def exitTimingStatement(self, ctx: qasm2Parser.TimingStatementContext):
        pass

    # Enter a parse tree produced by qasm2Parser#calibration.
    def enterCalibration(self, ctx: qasm2Parser.CalibrationContext):
        pass

    # Exit a parse tree produced by qasm2Parser#calibration.
    def exitCalibration(self, ctx: qasm2Parser.CalibrationContext):
        pass

    # Enter a parse tree produced by qasm2Parser#calibrationGrammarDeclaration.
    def enterCalibrationGrammarDeclaration(
        self, ctx: qasm2Parser.CalibrationGrammarDeclarationContext
    ):
        pass

    # Exit a parse tree produced by qasm2Parser#calibrationGrammarDeclaration.
    def exitCalibrationGrammarDeclaration(
        self, ctx: qasm2Parser.CalibrationGrammarDeclarationContext
    ):
        pass

    # Enter a parse tree produced by qasm2Parser#calibrationDefinition.
    def enterCalibrationDefinition(self, ctx: qasm2Parser.CalibrationDefinitionContext):
        pass

    # Exit a parse tree produced by qasm2Parser#calibrationDefinition.
    def exitCalibrationDefinition(self, ctx: qasm2Parser.CalibrationDefinitionContext):
        pass

    # Enter a parse tree produced by qasm2Parser#calibrationGrammar.
    def enterCalibrationGrammar(self, ctx: qasm2Parser.CalibrationGrammarContext):
        pass

    # Exit a parse tree produced by qasm2Parser#calibrationGrammar.
    def exitCalibrationGrammar(self, ctx: qasm2Parser.CalibrationGrammarContext):
        pass

    # Enter a parse tree produced by qasm2Parser#calibrationArgumentList.
    def enterCalibrationArgumentList(
        self, ctx: qasm2Parser.CalibrationArgumentListContext
    ):
        pass

    # Exit a parse tree produced by qasm2Parser#calibrationArgumentList.
    def exitCalibrationArgumentList(
        self, ctx: qasm2Parser.CalibrationArgumentListContext
    ):
        pass

    # Enter a parse tree produced by qasm2Parser#metaComment.
    def enterMetaComment(self, ctx: qasm2Parser.MetaCommentContext):
        pass

    # Exit a parse tree produced by qasm2Parser#metaComment.
    def exitMetaComment(self, ctx: qasm2Parser.MetaCommentContext):
        pass


del qasm2Parser
