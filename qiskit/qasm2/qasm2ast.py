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
qasm2ast.py

Created on Wed Apr 14 23:17:27 2021

The implementation "AST" (very loosely construed) to be generated
from parsing OpenQASM which is to be translated into QuantumCircuit.

Some of this code derives from https://github.com/jwoehr/nuqasm2
(Apache License 2.0 open source)

@author: jax jwoehr@softwoehr.com
"""

import datetime
import pprint

# from typing import List # Python doesn't allow inheritance from specialized types
from antlr4 import ParserRuleContext
from qiskit.qasm2.qasm2astelem import CodeBody, GateBody, SourceBody


# #########################
# Translation unit sections
# #########################


class TSect(dict):
    """Translation overhead section of translation unit"""

    def __init__(self, name) -> None:
        """Instance structures filled in by QasmTranslator"""
        super().__init__()
        self["name"] = name
        self["filepaths"] = []
        self["datetime_start"] = None
        self["datetime_finish"] = None

    def append_filepath(self, filepath: str) -> int:
        """Append to filepaths returning index of latest append"""
        idx = self.num_filepaths()
        self["filepaths"].append(filepath)
        return idx

    def num_filepaths(self) -> int:
        """Number of filepaths stored"""
        return len(self["filepaths"])

    def get_latest_file_index(self) -> int:
        """
        Get the last added file index assuming one has already been instanced.
        Returns -1 if none
        """
        return self.num_filepaths() - 1

    def set_datetime_start(self, datetime_start: datetime) -> None:
        """Record start of compilation"""
        self["datetime_start"] = datetime_start

    def get_datetime_start(self) -> datetime:
        """Get start of compilation"""
        return self["datetime_start"]

    def set_datetime_finish(self, datetime_finish: datetime) -> None:
        """Record end of compilation"""
        self["datetime_finish"] = datetime_finish

    def get_datetime_finish(self) -> datetime:
        """Get end of compilation"""
        return self["datetime_finish"]


class CSect(list):
    """Code section of translation unit, list of CodeBody"""


class GSect(list):
    """User gate definition section of translation unit, list of GateBody"""


class SSect(list):
    """Source code section, list of SourceBody"""


class Qasm2Stack(list):
    """Stack of parse tree nestings so we can track context at AST gen time"""

    def __init__(self, dbg: bool = False) -> None:
        super().__init__()
        self.dbg = dbg

    def push(self, ctx: ParserRuleContext) -> None:
        """Push the context we're in"""
        self.append(ctx)
        if self.dbg:
            pprint.pprint(self)

    def pop(self) -> ParserRuleContext:
        """Pop last context"""
        last = super().pop()
        if self.dbg:
            pprint.pprint(self)
        return last

    def peek(self, index: int = None) -> ParserRuleContext:
        """
        Peek nth element of the context stack from front.
        (0,1,2 ...)
        If None peek last.
        """
        ctx = None
        if index:
            ctx = self[index]
        else:
            ctx = self[len(self) - 1]
        return ctx

    def peek_back(self, index: int = None) -> ParserRuleContext:
        """
        Peek the nth previous element of the context stack.
        (... 2,1,0)
        If index == 0 peek last
        """
        return self[len(self) - index]


class Qasm2FilenumStack(list):
    """Stack tracking the filenums we service, list of int"""

    def __init__(self, dbg: bool = False) -> None:
        super().__init__()
        self.dbg = dbg

    def push(self, filenum: int) -> None:
        """Push the filenum we're in"""
        self.append(filenum)
        if self.dbg:
            pprint.pprint(self)

    def pop(self) -> int:
        """Pop last filenum"""
        filenum = super().pop()
        if self.dbg:
            pprint.pprint(self)
        return filenum

    def peek(self, index: int = None) -> ParserRuleContext:
        """
        Peek nth element of the filenum stack from front.
        (0,1,2 ...)
        If None peek last.
        """
        filenum = None
        if index:
            filenum = self[index]
        else:
            filenum = self[len(self) - 1]
        return filenum


class Qasm2AST(dict):
    """
    Contains the Sections that make up our AST derived from parsing Qasm
    and the Stack that tracks our state.
    """

    def __init__(self, name: str, *, dbg: list = None) -> None:
        """
        Instance Qasm2AST Structures
        Sections for translating
        State LIFO
        File include LIFO
        Scratch Space
        Debug flags
        """
        super().__init__()
        self.dbg = dbg if dbg else list()
        self["TSect"] = TSect(name)
        self["CSect"] = CSect()
        self["GSect"] = GSect()
        self["SSect"] = SSect()
        self.state = Qasm2Stack(dbg=self.dbg.count("Stack") > 0)
        self.file_lifo = Qasm2FilenumStack(dbg=self.dbg.count("Filenums") > 0)
        self.push_filenum(0)
        self.scratch = None

    def push(self, ctx: ParserRuleContext) -> None:
        """Push the context we're in"""
        self.state.push(ctx)

    def pop(self) -> ParserRuleContext:
        """Pop last context"""
        return self.state.pop()

    def peek(self, index: int = None) -> ParserRuleContext:
        """
        Peek nth element of the context stack from front.
        (0,1,2 ...)
        If None peek last.
        """
        return self.state.peek(index)

    def peek_back(self, index: int = None) -> ParserRuleContext:
        """
        Peek the nth previous element of the context stack.
        (... 2,1,0)
        If index == 0 peek last
        """
        return self.state.peek_back(index)

    def append_filepath(self, filepath) -> int:
        """
        Append a filepath bumping index.
        Return the index (to push onto the filenum lifo)
        """
        index = self["TSect"].append_filepath(filepath)
        return index

    def get_latest_file_index(self) -> int:
        """
        Return last-added index of source files,
        """
        return self["TSect"].get_latest_file_index()

    def append_code(self, code: CodeBody) -> None:
        """Add code body to the csect list of code bodies"""
        self["CSect"].append(code)

    def append_gatedef(self, gatedef: GateBody) -> None:
        """Append a gate definition to GSect list of gate definitions"""
        self["GSect"].append(gatedef)

    def prepend_gatedef(self, gatedef: GateBody) -> None:
        """Prepend a gate definition to GSect list of gate definitions"""
        self["GSect"].insert(0, gatedef)

    def find_latest_gatedef(self, op: str) -> dict:
        """
        Search GSect list of Gate definitions.
        It's a reverse-order list because we want the last superseding definition.

        Parameters
        ----------
        op : str
            Name of defined gate

        Returns
        -------
        dict
            GateBody of last definition (first in reversed GSect list) of op

        """
        latest = None
        for gatedef in self["GSect"]:
            if gatedef["op"] == op:
                latest = gatedef
                break
        return latest

    def append_source_body(self, source_body: SourceBody) -> None:
        """Append a fully-formed SourceBody to the SSect"""
        self["SSect"].append(source_body)

    def append_source(self, source: str) -> None:
        """
        Append source as SourceBody including latest file index.
        Assumes append_filepath() has already been called to bump index.
        """
        self.append_source_body(SourceBody(self.get_latest_file_index(), source))

    def set_datetime_start(self, datetime_start: datetime) -> None:
        """Record start of compilation"""
        self["TSect"].set_datetime_start(datetime_start)

    def get_datetime_start(self) -> datetime:
        """Get start of compilation"""
        return self["TSect"].get_datetime_start()

    def set_datetime_finish(self, datetime_finish: datetime) -> None:
        """Record end of compilation"""
        self["TSect"].set_datetime_finish(datetime_finish)

    def get_datetime_finish(self) -> datetime:
        """Get end of compilation"""
        return self["TSect"].get_datetime_finish()

    def push_filenum(self, filenum: int) -> None:
        """Push a current file number onto filenum lifo"""
        self.file_lifo.push(filenum)

    def pop_filenum(self) -> int:
        """Pop current file number from filenum lifo"""
        return self.file_lifo.pop()

    def peek_filenum(self) -> int:
        """Take a peek at the last (TOS) filenum"""
        return self.file_lifo.peek()
