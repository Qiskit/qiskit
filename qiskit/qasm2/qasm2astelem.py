"""
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

Created on Mon Apr 19 13:21:35 2021

Some of this code derives from https://github.com/jwoehr/nuqasm2
(Apache License 2.0 open source)

@author: jax jwoehr@softwoehr.com
"""
from typing import List
from antlr4 import ParserRuleContext
from qiskit.circuit import Gate


class SourceBody(dict):
    """Source code body with filenum of source file"""

    def __init__(self, filenum, source) -> None:
        """
        Instance structures filled in by QasmTranslator
        filenum ... index of filepath in TSect filepaths vector
        source ... source lines vector
        """
        super().__init__()
        self["filenum"] = filenum
        self["source"] = source


class GateBody(dict):
    """Gate definition entry in the GSect"""

    def __init__(
        self,
        filenum: int,
        linenum: int,
        ctx: ParserRuleContext,
        op: str = None,
        declaration: list = None,
        definition: list = None,
        parameter_list: list = None,
        target_list: list = None,
        gate: Gate = None,
    ) -> None:
        """Instance from qasm source code and parse into key:value pairs"""
        super().__init__()
        self["filenum"] = filenum
        self["linenum"] = linenum
        self["ctx"] = ctx
        self["op"] = op
        self["declaration"] = declaration
        self["parameter_list"] = parameter_list
        self["target_list"] = target_list
        self["definition"] = definition
        self["gate"] = gate


class CodeBody(dict):
    """
    Code statements in the CSect
    Superclass of all CodeBody classes
    Knows linenum, ctx, source, text
    linenum is source array line number
    source is source code
    """

    def __init__(self, filenum: int, linenum: int, ctx: ParserRuleContext) -> None:
        """Instance from qasm source code and parse into key:value pairs"""
        super().__init__()
        self["filenum"] = filenum
        self["linenum"] = linenum
        self["ctx"] = ctx


class CodeBodyQuantumDeclaration(CodeBody):
    """CSect element"""

    def __init__(
        self,
        filenum: int,
        linenum: int,
        ctx: ParserRuleContext,
        quantum_type=None,
        reg_name: str = None,
        reg_width: int = None,
    ) -> None:
        """Instance from qasm source code and parse into key:value pairs"""
        super().__init__(filenum, linenum, ctx)
        self["filenum"] = filenum
        self["linenum"] = linenum
        self["ctx"] = ctx
        self["quantum_type"] = quantum_type
        self["reg_name"] = reg_name
        self["reg_width"] = reg_width


class CodeBodyClassicalDeclaration(CodeBody):
    """CSect element"""

    def __init__(
        self,
        filenum: int,
        linenum: int,
        ctx: ParserRuleContext,
        bit_type=None,
        reg_name: str = None,
        reg_width: int = None,
    ) -> None:
        """Instance from qasm source code and parse into key:value pairs"""
        super().__init__(filenum, linenum, ctx)
        self["filenum"] = filenum
        self["linenum"] = linenum
        self["ctx"] = ctx
        self["bit_type"] = bit_type
        self["reg_name"] = reg_name
        self["reg_width"] = reg_width


class CodeBodyQuantumMeasurement(CodeBody):
    """CSect element"""

    def __init__(
        self,
        filenum: int,
        linenum: int,
        ctx: ParserRuleContext,
        index_identifier_list: List = None,
    ) -> None:
        """Instance from qasm source code and parse into key:value pairs"""
        super().__init__(filenum, linenum, ctx)
        self["filenum"] = filenum
        self["linenum"] = linenum
        self["ctx"] = ctx
        self["index_identifier_list"] = index_identifier_list


class CodeBodySubroutineCall(CodeBody):
    """CSect element"""

    def __init__(
        self,
        filenum: int,
        linenum: int,
        ctx: ParserRuleContext,
        op: str,
        parameter_list: List = None,
        target_list: List = None,
    ) -> None:
        """Instance from qasm source code and parse into key:value pairs"""
        super().__init__(filenum, linenum, ctx)
        self["filenum"] = filenum
        self["linenum"] = linenum
        self["ctx"] = ctx
        self["op"] = op
        self["parameter_list"] = parameter_list
        self["target_list"] = target_list


class CodeBodyQuantumBarrier(CodeBody):
    """CSect element"""

    def __init__(
        self,
        filenum: int,
        linenum: int,
        ctx: ParserRuleContext,
        index_identifier_list: List = None,
    ) -> None:
        """Instance from qasm source code and parse into key:value pairs"""
        super().__init__(filenum, linenum, ctx)
        self["filenum"] = filenum
        self["linenum"] = linenum
        self["ctx"] = ctx
        self["index_identifier_list"] = index_identifier_list


class CodeBodyBranchingStatement(CodeBody):
    """CSect element"""

    def __init__(
        self,
        filenum: int,
        linenum: int,
        ctx: ParserRuleContext,
        conditional: str,
        op: str,
        comparison_expression_list: list = None,
        parameter_list: list = None,
        target_list: list = None,
    ) -> None:
        """Instance from qasm source code and parse into key:value pairs"""
        super().__init__(filenum, linenum, ctx)
        self["filenum"] = filenum
        self["linenum"] = linenum
        self["ctx"] = ctx
        self["conditional"] = conditional
        self["op"] = op
        self["comparison_expression_list"] = comparison_expression_list
        self["parameter_list"] = parameter_list
        self["target_list"] = target_list


class CodeBodyMetaComment(CodeBody):
    """CSect element"""

    def __init__(
        self,
        filenum: int,
        linenum: int,
        ctx: ParserRuleContext,
        metacomment_list: List = None,
    ) -> None:
        """Instance from qasm source code and parse into key:value pairs"""
        super().__init__(filenum, linenum, ctx)
        self["filenum"] = filenum
        self["linenum"] = linenum
        self["ctx"] = ctx
        self["metacomment_list"] = metacomment_list
