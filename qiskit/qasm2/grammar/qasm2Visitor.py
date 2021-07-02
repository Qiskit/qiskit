# Generated from qasm2.g4 by ANTLR 4.9.2
from antlr4 import *

if __name__ is not None and "." in __name__:
    from .qasm2Parser import qasm2Parser
else:
    from qasm2Parser import qasm2Parser

# This class defines a complete generic visitor for a parse tree produced by qasm2Parser.


class qasm2Visitor(ParseTreeVisitor):

    # Visit a parse tree produced by qasm2Parser#program.
    def visitProgram(self, ctx: qasm2Parser.ProgramContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#header.
    def visitHeader(self, ctx: qasm2Parser.HeaderContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#version.
    def visitVersion(self, ctx: qasm2Parser.VersionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#include.
    def visitInclude(self, ctx: qasm2Parser.IncludeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#globalStatement.
    def visitGlobalStatement(self, ctx: qasm2Parser.GlobalStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#statement.
    def visitStatement(self, ctx: qasm2Parser.StatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#quantumDeclarationStatement.
    def visitQuantumDeclarationStatement(
        self, ctx: qasm2Parser.QuantumDeclarationStatementContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#classicalDeclarationStatement.
    def visitClassicalDeclarationStatement(
        self, ctx: qasm2Parser.ClassicalDeclarationStatementContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#classicalAssignment.
    def visitClassicalAssignment(self, ctx: qasm2Parser.ClassicalAssignmentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#assignmentStatement.
    def visitAssignmentStatement(self, ctx: qasm2Parser.AssignmentStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#returnSignature.
    def visitReturnSignature(self, ctx: qasm2Parser.ReturnSignatureContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#designator.
    def visitDesignator(self, ctx: qasm2Parser.DesignatorContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#doubleDesignator.
    def visitDoubleDesignator(self, ctx: qasm2Parser.DoubleDesignatorContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#identifierList.
    def visitIdentifierList(self, ctx: qasm2Parser.IdentifierListContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#association.
    def visitAssociation(self, ctx: qasm2Parser.AssociationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#quantumType.
    def visitQuantumType(self, ctx: qasm2Parser.QuantumTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#quantumDeclaration.
    def visitQuantumDeclaration(self, ctx: qasm2Parser.QuantumDeclarationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#quantumArgument.
    def visitQuantumArgument(self, ctx: qasm2Parser.QuantumArgumentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#quantumArgumentList.
    def visitQuantumArgumentList(self, ctx: qasm2Parser.QuantumArgumentListContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#bitType.
    def visitBitType(self, ctx: qasm2Parser.BitTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#singleDesignatorType.
    def visitSingleDesignatorType(self, ctx: qasm2Parser.SingleDesignatorTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#doubleDesignatorType.
    def visitDoubleDesignatorType(self, ctx: qasm2Parser.DoubleDesignatorTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#noDesignatorType.
    def visitNoDesignatorType(self, ctx: qasm2Parser.NoDesignatorTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#classicalType.
    def visitClassicalType(self, ctx: qasm2Parser.ClassicalTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#constantDeclaration.
    def visitConstantDeclaration(self, ctx: qasm2Parser.ConstantDeclarationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#singleDesignatorDeclaration.
    def visitSingleDesignatorDeclaration(
        self, ctx: qasm2Parser.SingleDesignatorDeclarationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#doubleDesignatorDeclaration.
    def visitDoubleDesignatorDeclaration(
        self, ctx: qasm2Parser.DoubleDesignatorDeclarationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#noDesignatorDeclaration.
    def visitNoDesignatorDeclaration(
        self, ctx: qasm2Parser.NoDesignatorDeclarationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#bitDeclaration.
    def visitBitDeclaration(self, ctx: qasm2Parser.BitDeclarationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#classicalDeclaration.
    def visitClassicalDeclaration(self, ctx: qasm2Parser.ClassicalDeclarationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#classicalTypeList.
    def visitClassicalTypeList(self, ctx: qasm2Parser.ClassicalTypeListContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#classicalArgument.
    def visitClassicalArgument(self, ctx: qasm2Parser.ClassicalArgumentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#classicalArgumentList.
    def visitClassicalArgumentList(self, ctx: qasm2Parser.ClassicalArgumentListContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#aliasStatement.
    def visitAliasStatement(self, ctx: qasm2Parser.AliasStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#indexIdentifier.
    def visitIndexIdentifier(self, ctx: qasm2Parser.IndexIdentifierContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#indexIdentifierList.
    def visitIndexIdentifierList(self, ctx: qasm2Parser.IndexIdentifierListContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#indexEqualsAssignmentList.
    def visitIndexEqualsAssignmentList(
        self, ctx: qasm2Parser.IndexEqualsAssignmentListContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#rangeDefinition.
    def visitRangeDefinition(self, ctx: qasm2Parser.RangeDefinitionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#quantumGateDefinition.
    def visitQuantumGateDefinition(self, ctx: qasm2Parser.QuantumGateDefinitionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#quantumGateSignature.
    def visitQuantumGateSignature(self, ctx: qasm2Parser.QuantumGateSignatureContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#quantumBlock.
    def visitQuantumBlock(self, ctx: qasm2Parser.QuantumBlockContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#quantumLoop.
    def visitQuantumLoop(self, ctx: qasm2Parser.QuantumLoopContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#quantumLoopBlock.
    def visitQuantumLoopBlock(self, ctx: qasm2Parser.QuantumLoopBlockContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#quantumStatement.
    def visitQuantumStatement(self, ctx: qasm2Parser.QuantumStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#quantumInstruction.
    def visitQuantumInstruction(self, ctx: qasm2Parser.QuantumInstructionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#quantumPhase.
    def visitQuantumPhase(self, ctx: qasm2Parser.QuantumPhaseContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#quantumMeasurement.
    def visitQuantumMeasurement(self, ctx: qasm2Parser.QuantumMeasurementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#quantumMeasurementAssignment.
    def visitQuantumMeasurementAssignment(
        self, ctx: qasm2Parser.QuantumMeasurementAssignmentContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#quantumBarrier.
    def visitQuantumBarrier(self, ctx: qasm2Parser.QuantumBarrierContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#quantumGateModifier.
    def visitQuantumGateModifier(self, ctx: qasm2Parser.QuantumGateModifierContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#quantumGateCall.
    def visitQuantumGateCall(self, ctx: qasm2Parser.QuantumGateCallContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#quantumGateName.
    def visitQuantumGateName(self, ctx: qasm2Parser.QuantumGateNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#unaryOperator.
    def visitUnaryOperator(self, ctx: qasm2Parser.UnaryOperatorContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#relationalOperator.
    def visitRelationalOperator(self, ctx: qasm2Parser.RelationalOperatorContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#logicalOperator.
    def visitLogicalOperator(self, ctx: qasm2Parser.LogicalOperatorContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#expressionStatement.
    def visitExpressionStatement(self, ctx: qasm2Parser.ExpressionStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#expression.
    def visitExpression(self, ctx: qasm2Parser.ExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#xOrExpression.
    def visitXOrExpression(self, ctx: qasm2Parser.XOrExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#bitAndExpression.
    def visitBitAndExpression(self, ctx: qasm2Parser.BitAndExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#bitShiftExpression.
    def visitBitShiftExpression(self, ctx: qasm2Parser.BitShiftExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#additiveExpression.
    def visitAdditiveExpression(self, ctx: qasm2Parser.AdditiveExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#multiplicativeExpression.
    def visitMultiplicativeExpression(
        self, ctx: qasm2Parser.MultiplicativeExpressionContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#unaryExpression.
    def visitUnaryExpression(self, ctx: qasm2Parser.UnaryExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#expressionTerminator.
    def visitExpressionTerminator(self, ctx: qasm2Parser.ExpressionTerminatorContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#incrementor.
    def visitIncrementor(self, ctx: qasm2Parser.IncrementorContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#builtInCall.
    def visitBuiltInCall(self, ctx: qasm2Parser.BuiltInCallContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#builtInMath.
    def visitBuiltInMath(self, ctx: qasm2Parser.BuiltInMathContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#castOperator.
    def visitCastOperator(self, ctx: qasm2Parser.CastOperatorContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#expressionList.
    def visitExpressionList(self, ctx: qasm2Parser.ExpressionListContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#booleanExpression.
    def visitBooleanExpression(self, ctx: qasm2Parser.BooleanExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#comparsionExpression.
    def visitComparsionExpression(self, ctx: qasm2Parser.ComparsionExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#equalsExpression.
    def visitEqualsExpression(self, ctx: qasm2Parser.EqualsExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#assignmentOperator.
    def visitAssignmentOperator(self, ctx: qasm2Parser.AssignmentOperatorContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#equalsAssignmentList.
    def visitEqualsAssignmentList(self, ctx: qasm2Parser.EqualsAssignmentListContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#membershipTest.
    def visitMembershipTest(self, ctx: qasm2Parser.MembershipTestContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#setDeclaration.
    def visitSetDeclaration(self, ctx: qasm2Parser.SetDeclarationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#programBlock.
    def visitProgramBlock(self, ctx: qasm2Parser.ProgramBlockContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#branchingStatement.
    def visitBranchingStatement(self, ctx: qasm2Parser.BranchingStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#loopSignature.
    def visitLoopSignature(self, ctx: qasm2Parser.LoopSignatureContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#loopStatement.
    def visitLoopStatement(self, ctx: qasm2Parser.LoopStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#controlDirectiveStatement.
    def visitControlDirectiveStatement(
        self, ctx: qasm2Parser.ControlDirectiveStatementContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#controlDirective.
    def visitControlDirective(self, ctx: qasm2Parser.ControlDirectiveContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#kernelDeclaration.
    def visitKernelDeclaration(self, ctx: qasm2Parser.KernelDeclarationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#kernelCall.
    def visitKernelCall(self, ctx: qasm2Parser.KernelCallContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#subroutineDefinition.
    def visitSubroutineDefinition(self, ctx: qasm2Parser.SubroutineDefinitionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#returnStatement.
    def visitReturnStatement(self, ctx: qasm2Parser.ReturnStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#subroutineBlock.
    def visitSubroutineBlock(self, ctx: qasm2Parser.SubroutineBlockContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#subroutineCall.
    def visitSubroutineCall(self, ctx: qasm2Parser.SubroutineCallContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#pragma.
    def visitPragma(self, ctx: qasm2Parser.PragmaContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#timingType.
    def visitTimingType(self, ctx: qasm2Parser.TimingTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#timingBox.
    def visitTimingBox(self, ctx: qasm2Parser.TimingBoxContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#timingTerminator.
    def visitTimingTerminator(self, ctx: qasm2Parser.TimingTerminatorContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#timingIdentifier.
    def visitTimingIdentifier(self, ctx: qasm2Parser.TimingIdentifierContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#timingInstructionName.
    def visitTimingInstructionName(self, ctx: qasm2Parser.TimingInstructionNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#timingInstruction.
    def visitTimingInstruction(self, ctx: qasm2Parser.TimingInstructionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#timingStatement.
    def visitTimingStatement(self, ctx: qasm2Parser.TimingStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#calibration.
    def visitCalibration(self, ctx: qasm2Parser.CalibrationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#calibrationGrammarDeclaration.
    def visitCalibrationGrammarDeclaration(
        self, ctx: qasm2Parser.CalibrationGrammarDeclarationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#calibrationDefinition.
    def visitCalibrationDefinition(self, ctx: qasm2Parser.CalibrationDefinitionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#calibrationGrammar.
    def visitCalibrationGrammar(self, ctx: qasm2Parser.CalibrationGrammarContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by qasm2Parser#calibrationArgumentList.
    def visitCalibrationArgumentList(
        self, ctx: qasm2Parser.CalibrationArgumentListContext
    ):
        return self.visitChildren(ctx)


del qasm2Parser
