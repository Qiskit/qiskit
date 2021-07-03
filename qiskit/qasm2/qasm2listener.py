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

"""
Created on Tue Mar 30 10:32:24 2021

The Qasm2 Listener based on the ANTLR 4 Grammar
We have used the Grammar being developed for OpenQASM 3,
the spec of which seems to be evolving as a pure superset of OpenQASM 2.

@author: jax jwoehr@softwoehr.com
"""

import os
from antlr4 import CommonTokenStream, InputStream, ParseTreeWalker
from qiskit.qasm2 import qasm2Lexer, qasm2Listener, qasm2Parser, QasmError
from qiskit.qasm2 import (
    Qasm2AST,
    GateBody,  # CodeBody, ourceBody,
    CodeBodyClassicalDeclaration,
    CodeBodyQuantumDeclaration,
    CodeBodyQuantumMeasurement,
    CodeBodySubroutineCall,
    CodeBodyQuantumBarrier,
    CodeBodyBranchingStatement,
    CodeBodyMetaComment,
)

CORE_LIBS_PATH = os.path.join(os.path.dirname(__file__), "libs")
CORE_LIBS = os.listdir(CORE_LIBS_PATH)


class Qasm2Listener(qasm2Listener):
    """
    Subclass of the ANTLR 4 generated Listener
    Generate an AST of the OpenQASM program to be
    translated by caller into instance of QuantumCircuit
    """

    def __init__(
        self,
        ast: Qasm2AST,
        input_src: str,
        debug_fh=None,
        include_path: str = ".",
        use_default_include: bool = True,
    ) -> None:
        """

        Parameters
        ----------
        ast : Qasm2AST
            a Qasm2AST instance (possibly already in use if this
            is a recursive entry due to an include).
        input_src : str
            input source code (from a simple `read` operation).
        debug_fh : resource, optional
            a debug fh (can be None) if output desired (e.g, os.stdout)
        include_path: str
            include path (colon-separated) to search for targets of include
            directive. The default is '.'
        use_default_include : bool, optional
            should "qelib.inc" be grabbed as a resource from Qiskit Terra
            instead of being sought in the include path as a file.
            The default is True.

        Returns
        -------
        None.

        """
        super().__init__()
        self.ast = ast
        self.input_src = input_src
        self.include_path = include_path
        self.use_default_include = use_default_include
        self.debug_fh = debug_fh
        self._last_error = QasmError("No Error")
        self.in_subroutine_call = False
        self.in_gate_definition = False
        self.in_branching_statement = False

    def dbg_write(self, text: str) -> None:
        """
        Printf debugging

        Parameters
        ----------
        text : str
            some text.

        Returns
        -------
        None.

        """
        if self.debug_fh:
            self.debug_fh.write(text)

    def do_ast(self) -> Qasm2AST:
        """
        Parse source from the textual source code
        self.input_src recursing into included files
        and populate the AST in self.ast.

        """
        _i = InputStream(self.input_src)
        lexer = qasm2Lexer(_i)
        stream = CommonTokenStream(lexer)
        stream.fill()
        parser = qasm2Parser(stream)
        tree = parser.program()
        walker = ParseTreeWalker()
        walker.walk(self, tree)
        return self.ast

    def find_include(self, filename: str) -> str:
        """
        Find include file in our search path

        Parameters
        ----------
        filename : str
            The include file we're looking for.

        Returns
        -------
        fullpath : str
            The full path to the file.

        """
        fullpath = None
        if os.path.isabs(filename):
            fullpath = filename
        else:
            for path in self.include_path.split(os.pathsep):
                candidate = os.path.join(path, filename)
                if os.path.exists(candidate):
                    fullpath = os.path.abspath(candidate)
        return fullpath

    def recurse_include(self, filename: str) -> None:
        """
        Step into an included file finding it in the include path.
        Magic: if it's the default include, fetch it as resource.
        Create a new Listener
        Pass it the extant AST
        Let it walk the included file and further populate the extant AST
        Parameters
        ----------
        filename : str
            sought file, can be fullor path relative to include path.

        Returns
        -------
        None

        Raises
        ------
        QasmError
            if include file not found

        """
        fullpath = None
        if filename in CORE_LIBS and self.use_default_include:
            fullpath = os.path.join(CORE_LIBS_PATH, filename)
        else:
            fullpath = self.find_include(filename)
            if not fullpath:
                raise QasmError(
                    "Include file {} not found in include path {}".format(
                        filename, self.include_path
                    )
                )
        _f = open(fullpath, "r")
        _src = _f.read()
        _f.close()
        self.ast.push_filenum(self.ast.append_filepath(fullpath))
        self.ast.append_source(_src)
        _q = Qasm2Listener(self.ast, _src, self.debug_fh)
        _q.do_ast()
        _ = self.ast.pop_filenum()

    # Enter a parse tree produced by qasm2Parser#program.
    def enterProgram(self, ctx: qasm2Parser.ProgramContext):
        self.ast.push(ctx)
        # self.dbg_write("Listener entering program")
        # self.dbg_write(ctx.toStringTree() + "\n")

    # Exit a parse tree produced by qasm2Parser#program.
    def exitProgram(self, ctx: qasm2Parser.ProgramContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#header.
    def enterHeader(self, ctx: qasm2Parser.HeaderContext):
        self.ast.push(ctx)
        self.dbg_write("Listener entering header")
        self.dbg_write(ctx.toStringTree() + "\n")

    # Exit a parse tree produced by qasm2Parser#header.
    def exitHeader(self, ctx: qasm2Parser.HeaderContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#version.
    def enterVersion(self, ctx: qasm2Parser.VersionContext):
        self.ast.push(ctx)
        self.dbg_write("Listener entering version")
        self.dbg_write(ctx.toStringTree() + "\n")
        self.dbg_write(ctx.getText() + "\n")

    # Exit a parse tree produced by qasm2Parser#version.
    def exitVersion(self, ctx: qasm2Parser.VersionContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#include.
    def enterInclude(self, ctx: qasm2Parser.IncludeContext):
        self.ast.push(ctx)
        self.dbg_write("Listener entering include")
        self.dbg_write(ctx.toStringTree() + "\n")
        self.dbg_write(ctx.getText() + "\n")
        for child in ctx.getChildren():
            self.dbg_write(child.getText() + "!")
        self.dbg_write("\n")
        include_statement = []
        for child in ctx.getChildren():
            include_statement.append(child.getText())
        include_file = include_statement[1].strip("\"' ")
        self.dbg_write("Include file: {}\n".format(include_file))
        self.recurse_include(include_file)

    # Exit a parse tree produced by qasm2Parser#include.
    def exitInclude(self, ctx: qasm2Parser.IncludeContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#ioIdentifier.
    def enterIoIdentifier(self, ctx: qasm2Parser.IoIdentifierContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#ioIdentifier.
    def exitIoIdentifier(self, ctx: qasm2Parser.IoIdentifierContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#io.
    def enterIo(self, ctx: qasm2Parser.IoContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#io.
    def exitIo(self, ctx: qasm2Parser.IoContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#globalStatement.
    def enterGlobalStatement(self, ctx: qasm2Parser.GlobalStatementContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#globalStatement.
    def exitGlobalStatement(self, ctx: qasm2Parser.GlobalStatementContext):
        if self.ast.scratch:
            if isinstance(self.ast.scratch, GateBody):
                self.ast.prepend_gatedef(self.ast.scratch)
            else:
                self.ast.append_code(self.ast.scratch)
            self.ast.scratch = None
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#statement.
    def enterStatement(self, ctx: qasm2Parser.StatementContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#statement.
    def exitStatement(self, ctx: qasm2Parser.StatementContext):
        if self.ast.scratch:
            self.ast.append_code(self.ast.scratch)
            self.ast.scratch = None
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#quantumDeclarationStatement.
    def enterQuantumDeclarationStatement(self, ctx: qasm2Parser.QuantumDeclarationStatementContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#quantumDeclarationStatement.
    def exitQuantumDeclarationStatement(self, ctx: qasm2Parser.QuantumDeclarationStatementContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#classicalDeclarationStatement.
    def enterClassicalDeclarationStatement(
        self, ctx: qasm2Parser.ClassicalDeclarationStatementContext
    ):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#classicalDeclarationStatement.
    def exitClassicalDeclarationStatement(
        self, ctx: qasm2Parser.ClassicalDeclarationStatementContext
    ):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#classicalAssignment.
    def enterClassicalAssignment(self, ctx: qasm2Parser.ClassicalAssignmentContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#classicalAssignment.
    def exitClassicalAssignment(self, ctx: qasm2Parser.ClassicalAssignmentContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#assignmentStatement.
    def enterAssignmentStatement(self, ctx: qasm2Parser.AssignmentStatementContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#assignmentStatement.
    def exitAssignmentStatement(self, ctx: qasm2Parser.AssignmentStatementContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#returnSignature.
    def enterReturnSignature(self, ctx: qasm2Parser.ReturnSignatureContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#returnSignature.
    def exitReturnSignature(self, ctx: qasm2Parser.ReturnSignatureContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#designator.
    def enterDesignator(self, ctx: qasm2Parser.DesignatorContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#designator.
    def exitDesignator(self, ctx: qasm2Parser.DesignatorContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#doubleDesignator.
    def enterDoubleDesignator(self, ctx: qasm2Parser.DoubleDesignatorContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#doubleDesignator.
    def exitDoubleDesignator(self, ctx: qasm2Parser.DoubleDesignatorContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#identifierList.
    def enterIdentifierList(self, ctx: qasm2Parser.IdentifierListContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#identifierList.
    def exitIdentifierList(self, ctx: qasm2Parser.IdentifierListContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#quantumDeclaration.
    def enterQuantumDeclaration(self, ctx: qasm2Parser.QuantumDeclarationContext):
        self.ast.push(ctx)
        self.ast.scratch = CodeBodyQuantumDeclaration(
            self.ast.peek_filenum(),
            ctx.start.line,
            ctx,
            ctx.getChild(0).getText(),
            ctx.getChild(1).getText(),
            int(ctx.getChild(2).getText().strip("[]")),
        )

    # Exit a parse tree produced by qasm2Parser#quantumDeclaration.
    def exitQuantumDeclaration(self, ctx: qasm2Parser.QuantumDeclarationContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#quantumArgument.
    def enterQuantumArgument(self, ctx: qasm2Parser.QuantumArgumentContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#quantumArgument.
    def exitQuantumArgument(self, ctx: qasm2Parser.QuantumArgumentContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#quantumArgumentList.
    def enterQuantumArgumentList(self, ctx: qasm2Parser.QuantumArgumentListContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#quantumArgumentList.
    def exitQuantumArgumentList(self, ctx: qasm2Parser.QuantumArgumentListContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#bitType.
    def enterBitType(self, ctx: qasm2Parser.BitTypeContext):
        self.ast.push(ctx)
        if self.ast.scratch:  # this test is probably a temporary expedient
            self.ast.scratch["bit_type"] = ctx.getText()

    # Exit a parse tree produced by qasm2Parser#bitType.
    def exitBitType(self, ctx: qasm2Parser.BitTypeContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#singleDesignatorType.
    def enterSingleDesignatorType(self, ctx: qasm2Parser.SingleDesignatorTypeContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#singleDesignatorType.
    def exitSingleDesignatorType(self, ctx: qasm2Parser.SingleDesignatorTypeContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#doubleDesignatorType.
    def enterDoubleDesignatorType(self, ctx: qasm2Parser.DoubleDesignatorTypeContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#doubleDesignatorType.
    def exitDoubleDesignatorType(self, ctx: qasm2Parser.DoubleDesignatorTypeContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#noDesignatorType.
    def enterNoDesignatorType(self, ctx: qasm2Parser.NoDesignatorTypeContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#noDesignatorType.
    def exitNoDesignatorType(self, ctx: qasm2Parser.NoDesignatorTypeContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#classicalType.
    def enterClassicalType(self, ctx: qasm2Parser.ClassicalTypeContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#classicalType.
    def exitClassicalType(self, ctx: qasm2Parser.ClassicalTypeContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#constantDeclaration.
    def enterConstantDeclaration(self, ctx: qasm2Parser.ConstantDeclarationContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#constantDeclaration.
    def exitConstantDeclaration(self, ctx: qasm2Parser.ConstantDeclarationContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#singleDesignatorDeclaration.
    def enterSingleDesignatorDeclaration(self, ctx: qasm2Parser.SingleDesignatorDeclarationContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#singleDesignatorDeclaration.
    def exitSingleDesignatorDeclaration(self, ctx: qasm2Parser.SingleDesignatorDeclarationContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#doubleDesignatorDeclaration.
    def enterDoubleDesignatorDeclaration(self, ctx: qasm2Parser.DoubleDesignatorDeclarationContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#doubleDesignatorDeclaration.
    def exitDoubleDesignatorDeclaration(self, ctx: qasm2Parser.DoubleDesignatorDeclarationContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#noDesignatorDeclaration.
    def enterNoDesignatorDeclaration(self, ctx: qasm2Parser.NoDesignatorDeclarationContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#noDesignatorDeclaration.
    def exitNoDesignatorDeclaration(self, ctx: qasm2Parser.NoDesignatorDeclarationContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#bitDeclaration.
    def enterBitDeclaration(self, ctx: qasm2Parser.BitDeclarationContext):
        self.ast.push(ctx)
        if isinstance(self.ast.scratch, CodeBodyClassicalDeclaration):
            self.ast.scratch["bit_type"] = ctx.getChild(0).getText()
            self.ast.scratch["reg_name"] = ctx.getChild(1).getText()
            self.ast.scratch["reg_width"] = int(ctx.getChild(2).getText().strip("[]"))

    # Exit a parse tree produced by qasm2Parser#bitDeclaration.
    def exitBitDeclaration(self, ctx: qasm2Parser.BitDeclarationContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#classicalDeclaration.
    def enterClassicalDeclaration(self, ctx: qasm2Parser.ClassicalDeclarationContext):
        self.ast.push(ctx)
        self.ast.scratch = CodeBodyClassicalDeclaration(
            self.ast.peek_filenum(), ctx.start.line, ctx, None, None, None
        )

    # Exit a parse tree produced by qasm2Parser#classicalDeclaration.
    def exitClassicalDeclaration(self, ctx: qasm2Parser.ClassicalDeclarationContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#classicalTypeList.
    def enterClassicalTypeList(self, ctx: qasm2Parser.ClassicalTypeListContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#classicalTypeList.
    def exitClassicalTypeList(self, ctx: qasm2Parser.ClassicalTypeListContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#classicalArgument.
    def enterClassicalArgument(self, ctx: qasm2Parser.ClassicalArgumentContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#classicalArgument.
    def exitClassicalArgument(self, ctx: qasm2Parser.ClassicalArgumentContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#classicalArgumentList.
    def enterClassicalArgumentList(self, ctx: qasm2Parser.ClassicalArgumentListContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#classicalArgumentList.
    def exitClassicalArgumentList(self, ctx: qasm2Parser.ClassicalArgumentListContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#aliasStatement.
    def enterAliasStatement(self, ctx: qasm2Parser.AliasStatementContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#aliasStatement.
    def exitAliasStatement(self, ctx: qasm2Parser.AliasStatementContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#indexIdentifier.
    def enterIndexIdentifier(self, ctx: qasm2Parser.IndexIdentifierContext):
        self.ast.push(ctx)
        if (
            self.ast.scratch
            and isinstance(self.ast.scratch, CodeBodyQuantumBarrier)
            or isinstance(self.ast.scratch, CodeBodyQuantumMeasurement)
        ):
            # this test is probably a temporary expedient
            if isinstance(self.ast.scratch["index_identifier_list"], list):
                self.ast.scratch["index_identifier_list"].append(ctx.getText())

        elif isinstance(self.ast.scratch, CodeBodySubroutineCall) or isinstance(
            self.ast.scratch, CodeBodyBranchingStatement
        ):
            self.ast.scratch["target_list"].append(ctx.getText())

    # Exit a parse tree produced by qasm2Parser#indexIdentifier.
    def exitIndexIdentifier(self, ctx: qasm2Parser.IndexIdentifierContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#indexIdentifierList.
    def enterIndexIdentifierList(self, ctx: qasm2Parser.IndexIdentifierListContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#indexIdentifierList.
    def exitIndexIdentifierList(self, ctx: qasm2Parser.IndexIdentifierListContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#rangeDefinition.
    def enterRangeDefinition(self, ctx: qasm2Parser.RangeDefinitionContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#rangeDefinition.
    def exitRangeDefinition(self, ctx: qasm2Parser.RangeDefinitionContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#quantumGateDefinition.
    def enterQuantumGateDefinition(self, ctx: qasm2Parser.QuantumGateDefinitionContext):
        self.ast.push(ctx)
        self.in_gate_definition = True
        self.ast.scratch = GateBody(
            self.ast.peek_filenum(),
            ctx.start.line,
            ctx,
            None,
            list(),
            list(),
            list(),
            list(),
        )
        self.dbg_write(
            "enterQuantumGateDefinition on line {} type start {}\n".format(
                ctx.start.line, str(type(ctx.start))
            )
        )

    # Exit a parse tree produced by qasm2Parser#quantumGateDefinition.
    def exitQuantumGateDefinition(self, ctx: qasm2Parser.QuantumGateDefinitionContext):
        self.in_gate_definition = False
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#quantumGateSignature.
    def enterQuantumGateSignature(self, ctx: qasm2Parser.QuantumGateSignatureContext):
        self.ast.push(ctx)
        if self.ast.scratch:
            for child in ctx.getChildren():
                _txt = child.getText()
                self.ast.scratch["declaration"].append(_txt)
            self.ast.scratch["op"] = self.ast.scratch["declaration"][0]
            if self.ast.scratch["declaration"][1] == "(":
                self.ast.scratch["parameter_list"] = self.ast.scratch["declaration"][2].split(",")
                self.ast.scratch["target_list"] = self.ast.scratch["declaration"][4].split(",")
            else:
                self.ast.scratch["target_list"] = self.ast.scratch["declaration"][1].split(",")

    # Exit a parse tree produced by qasm2Parser#quantumGateSignature.
    def exitQuantumGateSignature(self, ctx: qasm2Parser.QuantumGateSignatureContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#quantumGateName.
    def enterQuantumGateName(self, ctx: qasm2Parser.QuantumGateNameContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#quantumGateName.
    def exitQuantumGateName(self, ctx: qasm2Parser.QuantumGateNameContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#quantumBlock.
    def enterQuantumBlock(self, ctx: qasm2Parser.QuantumBlockContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#quantumBlock.
    def exitQuantumBlock(self, ctx: qasm2Parser.QuantumBlockContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#quantumLoop.
    def enterQuantumLoop(self, ctx: qasm2Parser.QuantumLoopContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#quantumLoop.
    def exitQuantumLoop(self, ctx: qasm2Parser.QuantumLoopContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#quantumLoopBlock.
    def enterQuantumLoopBlock(self, ctx: qasm2Parser.QuantumLoopBlockContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#quantumLoopBlock.
    def exitQuantumLoopBlock(self, ctx: qasm2Parser.QuantumLoopBlockContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#quantumStatement.
    def enterQuantumStatement(self, ctx: qasm2Parser.QuantumStatementContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#quantumStatement.
    def exitQuantumStatement(self, ctx: qasm2Parser.QuantumStatementContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#quantumInstruction.
    def enterQuantumInstruction(self, ctx: qasm2Parser.QuantumInstructionContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#quantumInstruction.
    def exitQuantumInstruction(self, ctx: qasm2Parser.QuantumInstructionContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#quantumPhase.
    def enterQuantumPhase(self, ctx: qasm2Parser.QuantumPhaseContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#quantumPhase.
    def exitQuantumPhase(self, ctx: qasm2Parser.QuantumPhaseContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#quantumReset.
    def enterQuantumReset(self, ctx: qasm2Parser.QuantumResetContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#quantumReset.
    def exitQuantumReset(self, ctx: qasm2Parser.QuantumResetContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#quantumMeasurement.
    def enterQuantumMeasurement(self, ctx: qasm2Parser.QuantumMeasurementContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#quantumMeasurement.
    def exitQuantumMeasurement(self, ctx: qasm2Parser.QuantumMeasurementContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#quantumMeasurementAssignment.
    def enterQuantumMeasurementAssignment(
        self, ctx: qasm2Parser.QuantumMeasurementAssignmentContext
    ):
        self.ast.push(ctx)
        self.ast.scratch = CodeBodyQuantumMeasurement(
            self.ast.peek_filenum(), ctx.start.line, ctx, list()
        )

    # Exit a parse tree produced by qasm2Parser#quantumMeasurementAssignment.
    def exitQuantumMeasurementAssignment(
        self, ctx: qasm2Parser.QuantumMeasurementAssignmentContext
    ):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#quantumBarrier.
    def enterQuantumBarrier(self, ctx: qasm2Parser.QuantumBarrierContext):
        self.ast.push(ctx)
        self.ast.scratch = CodeBodyQuantumBarrier(
            self.ast.peek_filenum(), ctx.start.line, ctx, list()
        )

    # Exit a parse tree produced by qasm2Parser#quantumBarrier.
    def exitQuantumBarrier(self, ctx: qasm2Parser.QuantumBarrierContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#quantumGateModifier.
    def enterQuantumGateModifier(self, ctx: qasm2Parser.QuantumGateModifierContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#quantumGateModifier.
    def exitQuantumGateModifier(self, ctx: qasm2Parser.QuantumGateModifierContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#powModifier.
    def enterPowModifier(self, ctx: qasm2Parser.PowModifierContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#powModifier.
    def exitPowModifier(self, ctx: qasm2Parser.PowModifierContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#ctrlModifier.
    def enterCtrlModifier(self, ctx: qasm2Parser.CtrlModifierContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#ctrlModifier.
    def exitCtrlModifier(self, ctx: qasm2Parser.CtrlModifierContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#quantumGateCall.
    def enterQuantumGateCall(self, ctx: qasm2Parser.QuantumGateCallContext):
        self.ast.push(ctx)
        if isinstance(self.ast.scratch, GateBody):
            _call = dict()
            _txt = list()
            for child in ctx.getChildren():
                _txt.append(child.getText())
            _call["op"] = _txt[0]
            if _txt[1] == "(":
                _call["parameter_list"] = _txt[2].split(",")
                _call["target_list"] = _txt[4].split(",")
            else:
                _call["parameter_list"] = list()
                _call["target_list"] = _txt[1].split(",")
            self.ast.scratch["definition"].append(_call)

    # Exit a parse tree produced by qasm2Parser#quantumGateCall.
    def exitQuantumGateCall(self, ctx: qasm2Parser.QuantumGateCallContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#unaryOperator.
    def enterUnaryOperator(self, ctx: qasm2Parser.UnaryOperatorContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#unaryOperator.
    def exitUnaryOperator(self, ctx: qasm2Parser.UnaryOperatorContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#relationalOperator.
    def enterComparisonOperator(self, ctx: qasm2Parser.ComparisonOperatorContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#relationalOperator.
    def exitComparisonOperator(self, ctx: qasm2Parser.ComparisonOperatorContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#equalityOperator.
    def enterEqualityOperator(self, ctx: qasm2Parser.EqualityOperatorContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#equalityOperator.
    def exitEqualityOperator(self, ctx: qasm2Parser.EqualityOperatorContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#logicalOperator.
    def enterLogicalOperator(self, ctx: qasm2Parser.LogicalOperatorContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#logicalOperator.
    def exitLogicalOperator(self, ctx: qasm2Parser.LogicalOperatorContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#expressionStatement.
    def enterExpressionStatement(self, ctx: qasm2Parser.ExpressionStatementContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#expressionStatement.
    def exitExpressionStatement(self, ctx: qasm2Parser.ExpressionStatementContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#expression.
    def enterExpression(self, ctx: qasm2Parser.ExpressionContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#expression.
    def exitExpression(self, ctx: qasm2Parser.ExpressionContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#logicalAndExpression.
    def enterLogicalAndExpression(self, ctx: qasm2Parser.LogicalAndExpressionContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#logicalAndExpression.
    def exitLogicalAndExpression(self, ctx: qasm2Parser.LogicalAndExpressionContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#bitOrExpression.
    def enterBitOrExpression(self, ctx: qasm2Parser.BitOrExpressionContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#bitOrExpression.
    def exitBitOrExpression(self, ctx: qasm2Parser.BitOrExpressionContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#xOrExpression.
    def enterXOrExpression(self, ctx: qasm2Parser.XOrExpressionContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#xOrExpression.
    def exitXOrExpression(self, ctx: qasm2Parser.XOrExpressionContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#bitAndExpression.
    def enterBitAndExpression(self, ctx: qasm2Parser.BitAndExpressionContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#bitAndExpression.
    def exitBitAndExpression(self, ctx: qasm2Parser.BitAndExpressionContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#equalityExpression.
    def enterEqualityExpression(self, ctx: qasm2Parser.EqualityExpressionContext):
        self.ast.push(ctx)
        if self.in_branching_statement and ctx.getChildCount() >= 3:
            _comparison_expression_list = []
            for i in ctx.getChildren():
                _comparison_expression_list.append(i.getText())
            self.ast.scratch["comparison_expression_list"] = _comparison_expression_list

    # Exit a parse tree produced by qasm2Parser#equalityExpression.
    def exitEqualityExpression(self, ctx: qasm2Parser.EqualityExpressionContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#ComparisonExpression.
    def enterComparisonExpression(self, ctx: qasm2Parser.ComparisonExpressionContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#comparisonExpression.
    def exitComparisonExpression(self, ctx: qasm2Parser.ComparisonExpressionContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#bitShiftExpression.
    def enterBitShiftExpression(self, ctx: qasm2Parser.BitShiftExpressionContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#bitShiftExpression.
    def exitBitShiftExpression(self, ctx: qasm2Parser.BitShiftExpressionContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#additiveExpression.
    def enterAdditiveExpression(self, ctx: qasm2Parser.AdditiveExpressionContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#additiveExpression.
    def exitAdditiveExpression(self, ctx: qasm2Parser.AdditiveExpressionContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#multiplicativeExpression.
    def enterMultiplicativeExpression(self, ctx: qasm2Parser.MultiplicativeExpressionContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#multiplicativeExpression.
    def exitMultiplicativeExpression(self, ctx: qasm2Parser.MultiplicativeExpressionContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#unaryExpression.
    def enterUnaryExpression(self, ctx: qasm2Parser.UnaryExpressionContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#unaryExpression.
    def exitUnaryExpression(self, ctx: qasm2Parser.UnaryExpressionContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#powerExpression.
    def enterPowerExpression(self, ctx: qasm2Parser.PowerExpressionContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#powerExpression.
    def exitPowerExpression(self, ctx: qasm2Parser.PowerExpressionContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#expressionTerminator.
    def enterExpressionTerminator(self, ctx: qasm2Parser.ExpressionTerminatorContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#expressionTerminator.
    def exitExpressionTerminator(self, ctx: qasm2Parser.ExpressionTerminatorContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#incrementor.
    def enterIncrementor(self, ctx: qasm2Parser.IncrementorContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#incrementor.
    def exitIncrementor(self, ctx: qasm2Parser.IncrementorContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#builtInCall.
    def enterBuiltInCall(self, ctx: qasm2Parser.BuiltInCallContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#builtInCall.
    def exitBuiltInCall(self, ctx: qasm2Parser.BuiltInCallContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#builtInMath.
    def enterBuiltInMath(self, ctx: qasm2Parser.BuiltInMathContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#builtInMath.
    def exitBuiltInMath(self, ctx: qasm2Parser.BuiltInMathContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#castOperator.
    def enterCastOperator(self, ctx: qasm2Parser.CastOperatorContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#castOperator.
    def exitCastOperator(self, ctx: qasm2Parser.CastOperatorContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#expressionList.
    def enterExpressionList(self, ctx: qasm2Parser.ExpressionListContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#expressionList.
    def exitExpressionList(self, ctx: qasm2Parser.ExpressionListContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#equalsExpression.
    def enterEqualsExpression(self, ctx: qasm2Parser.EqualsExpressionContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#equalsExpression.
    def exitEqualsExpression(self, ctx: qasm2Parser.EqualsExpressionContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#assignmentOperator.
    def enterAssignmentOperator(self, ctx: qasm2Parser.AssignmentOperatorContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#assignmentOperator.
    def exitAssignmentOperator(self, ctx: qasm2Parser.AssignmentOperatorContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#setDeclaration.
    def enterSetDeclaration(self, ctx: qasm2Parser.SetDeclarationContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#setDeclaration.
    def exitSetDeclaration(self, ctx: qasm2Parser.SetDeclarationContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#programBlock.
    def enterProgramBlock(self, ctx: qasm2Parser.ProgramBlockContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#programBlock.
    def exitProgramBlock(self, ctx: qasm2Parser.ProgramBlockContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#branchingStatement.
    def enterBranchingStatement(self, ctx: qasm2Parser.BranchingStatementContext):
        self.ast.push(ctx)
        self.in_branching_statement = True
        self.ast.scratch = CodeBodyBranchingStatement(
            self.ast.peek_filenum(),
            ctx.start.line,
            ctx,
            ctx.getChild(0).getText(),
            None,
            list(),
            list(),
            list(),
        )

    # Exit a parse tree produced by qasm2Parser#branchingStatement.
    def exitBranchingStatement(self, ctx: qasm2Parser.BranchingStatementContext):
        self.in_branching_statement = False
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#loopSignature.
    def enterLoopSignature(self, ctx: qasm2Parser.LoopSignatureContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#loopSignature.
    def exitLoopSignature(self, ctx: qasm2Parser.LoopSignatureContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#loopStatement.
    def enterLoopStatement(self, ctx: qasm2Parser.LoopStatementContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#loopStatement.
    def exitLoopStatement(self, ctx: qasm2Parser.LoopStatementContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#endStatement.
    def enterEndStatement(self, ctx: qasm2Parser.EndStatementContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#endStatement.
    def exitEndStatement(self, ctx: qasm2Parser.EndStatementContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#returnStatement.
    def enterReturnStatement(self, ctx: qasm2Parser.ReturnStatementContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#returnStatement.
    def exitReturnStatement(self, ctx: qasm2Parser.ReturnStatementContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#controlDirective.
    def enterControlDirective(self, ctx: qasm2Parser.ControlDirectiveContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#controlDirective.
    def exitControlDirective(self, ctx: qasm2Parser.ControlDirectiveContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#externDeclaration.
    def enterExternDeclaration(self, ctx: qasm2Parser.ExternDeclarationContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#externDeclaration.
    def exitExternDeclaration(self, ctx: qasm2Parser.ExternDeclarationContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#externCall.
    def enterExternCall(self, ctx: qasm2Parser.ExternCallContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#externCall.
    def exitExternCall(self, ctx: qasm2Parser.ExternCallContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#subroutineDefinition.
    def enterSubroutineDefinition(self, ctx: qasm2Parser.SubroutineDefinitionContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#subroutineDefinition.
    def exitSubroutineDefinition(self, ctx: qasm2Parser.SubroutineDefinitionContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#subroutineBlock.
    def enterSubroutineBlock(self, ctx: qasm2Parser.SubroutineBlockContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#subroutineBlock.
    def exitSubroutineBlock(self, ctx: qasm2Parser.SubroutineBlockContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#subroutineCall.
    def enterSubroutineCall(self, ctx: qasm2Parser.SubroutineCallContext):
        self.ast.push(ctx)
        self.in_subroutine_call = True
        if not self.in_gate_definition and not self.in_branching_statement:
            self.ast.scratch = CodeBodySubroutineCall(
                self.ast.peek_filenum(),
                ctx.start.line,
                ctx,
                ctx.getChild(0).getText(),
                list(),
                list(),
            )
            if ctx.getChildCount() > 2 and ctx.getChild(1).getText() == "(":
                _txt = ctx.getChild(2).getText()
                self.ast.scratch["parameter_list"] = _txt.split(",")
        if self.in_branching_statement:
            self.ast.scratch["op"] = ctx.getChild(0).getText()
            if ctx.getChildCount() > 2 and ctx.getChild(1).getText() == "(":
                _txt = ctx.getChild(2).getText()
                self.ast.scratch["parameter_list"] = _txt.split(",")

    # Exit a parse tree produced by qasm2Parser#subroutineCall.
    def exitSubroutineCall(self, ctx: qasm2Parser.SubroutineCallContext):
        self.in_subroutine_call = False
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#pragma.
    def enterPragma(self, ctx: qasm2Parser.PragmaContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#pragma.
    def exitPragma(self, ctx: qasm2Parser.PragmaContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#timingType.
    def enterTimingType(self, ctx: qasm2Parser.TimingTypeContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#timingType.
    def exitTimingType(self, ctx: qasm2Parser.TimingTypeContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#timingBox.
    def enterTimingBox(self, ctx: qasm2Parser.TimingBoxContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#timingBox.
    def exitTimingBox(self, ctx: qasm2Parser.TimingBoxContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#timingIdentifier.
    def enterTimingIdentifier(self, ctx: qasm2Parser.TimingIdentifierContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#timingIdentifier.
    def exitTimingIdentifier(self, ctx: qasm2Parser.TimingIdentifierContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#timingInstructionName.
    def enterTimingInstructionName(self, ctx: qasm2Parser.TimingInstructionNameContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#timingInstructionName.
    def exitTimingInstructionName(self, ctx: qasm2Parser.TimingInstructionNameContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#timingInstruction.
    def enterTimingInstruction(self, ctx: qasm2Parser.TimingInstructionContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#timingInstruction.
    def exitTimingInstruction(self, ctx: qasm2Parser.TimingInstructionContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#timingStatement.
    def enterTimingStatement(self, ctx: qasm2Parser.TimingStatementContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#timingStatement.
    def exitTimingStatement(self, ctx: qasm2Parser.TimingStatementContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#calibration.
    def enterCalibration(self, ctx: qasm2Parser.CalibrationContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#calibration.
    def exitCalibration(self, ctx: qasm2Parser.CalibrationContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#calibrationGrammarDeclaration.
    def enterCalibrationGrammarDeclaration(
        self, ctx: qasm2Parser.CalibrationGrammarDeclarationContext
    ):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#calibrationGrammarDeclaration.
    def exitCalibrationGrammarDeclaration(
        self, ctx: qasm2Parser.CalibrationGrammarDeclarationContext
    ):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#calibrationDefinition.
    def enterCalibrationDefinition(self, ctx: qasm2Parser.CalibrationDefinitionContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#calibrationDefinition.
    def exitCalibrationDefinition(self, ctx: qasm2Parser.CalibrationDefinitionContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#calibrationGrammar.
    def enterCalibrationGrammar(self, ctx: qasm2Parser.CalibrationGrammarContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#calibrationGrammar.
    def exitCalibrationGrammar(self, ctx: qasm2Parser.CalibrationGrammarContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#calibrationArgumentList.
    def enterCalibrationArgumentList(self, ctx: qasm2Parser.CalibrationArgumentListContext):
        self.ast.push(ctx)

    # Exit a parse tree produced by qasm2Parser#calibrationArgumentList.
    def exitCalibrationArgumentList(self, ctx: qasm2Parser.CalibrationArgumentListContext):
        _ = self.ast.pop()

    # Enter a parse tree produced by qasm2Parser#metaComment.
    def enterMetaComment(self, ctx: qasm2Parser.MetaCommentContext):
        self.ast.push(ctx)
        self.ast.scratch = CodeBodyMetaComment(self.ast.peek_filenum(), ctx.start.line, ctx, list())
        for child in ctx.getChildren():
            self.ast.scratch["metacomment_list"].append(child.getText())

    # Exit a parse tree produced by qasm2Parser#metaComment.
    def exitMetaComment(self, ctx: qasm2Parser.MetaCommentContext):
        if self.ast.scratch:
            self.ast.append_code(self.ast.scratch)
            self.ast.scratch = None
        _ = self.ast.pop()
