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

Created on Wed Apr 28 08:25:12 2021

@author: jax jwoehr@softwoehr.com
"""

import datetime
import io
import os
import sys
import pstats
import traceback
from typing import Tuple

# from qiskit import QuantumCircuit
from qiskit.qasm2 import Qasm2Listener, Qasm2AST
from qiskit.qasm2 import QasmError
from qiskit.qasm2.qasm2translator import Qasm2Translator


class Qasm2Loader:
    """
    Qasm2Loader is the translator of OpenQASM 2 source code to QuantumCircuit.
    """

    def __init__(
        self,
        *,
        qasm_str: str = None,
        qasm_filepath: str = None,
        name: str = None,
        include_path: str = ".",
        default_include: bool = True,
    ) -> None:
        """
        Init the loader with all factors for a load.

        Parameters
        ----------
        qasm_str : str, optional
            OpenQASM source text. Mutually exclusive with qasm_filepath.
            The default is None.
        qasm_filepath : str, optional
            OpenQASM source filepath. Mutually exclusive with qasm_str.
            The default is 'None'.
        name : str, optional
            Name for the circuit
        include_path : str, optional
            Colon-separated path in which to seek user-supplied include files.
            TThe default is '.' the current directory.
        default_include : bool, optional
            If true, the default include file is the one included when the
            un-pathed name 'qelib1.inc' is encountered as an include.
            The default is True.

        Raises
        ------
        QasmError
            If both qasm_str and qasm_filepath are supplied.

        Returns
        -------
        None.

        """
        if qasm_str is not None and qasm_filepath is not None:
            raise QasmError("Only one or the other is allowed of qasm_str and qasm_filepath")
        self.qasm_str = qasm_str
        self.qasm_filepath = qasm_filepath
        self.name = name
        self.include_path = include_path
        self.default_include = default_include
        self.ast = None
        self.qt = None  # pylint: disable=invalid-name
        self.qc = None

    def ast_qasm_str(self, qasm_str: str, filename: str, name: str, dbg: list = None) -> Qasm2AST:
        """
        Translate OpenQASM source string to AST

        Parameters
        ----------
        qasm_str : str
            OpenQASM source as a string.
        filename : str
            Nominal "filename" to assign to the string.
        name : str
            Name for the circuit.
        dbg : list, optional
            The default is None.

        Returns
        -------
        Qasm2AST
            The ``Qasm2AST`` resulting from the ``qasm_str`` provided.

        """
        ast = Qasm2AST(name, dbg=dbg)
        ast.append_filepath(filename)
        ast.append_source(qasm_str)
        ast.set_datetime_start(datetime.datetime.now().isoformat())
        q2_listener = Qasm2Listener(
            ast,
            qasm_str,
            debug_fh=None,
            include_path=self.include_path,
            use_default_include=self.default_include,
        )
        q2_listener.do_ast()
        ast.set_datetime_finish(datetime.datetime.now().isoformat())
        return ast

    def ast_qasm_file(self, filepath: str, name: str = None, dbg: list = None) -> Qasm2AST:
        """
        Translate OpenQASM file to AST

        Parameters
        ----------
        filepath : str
            Path of OpenQASM source file to load, relative or absolute.
        name : str, optional
            Name for the circuit. The default is None.
        dbg : list, optional
            "Magic" debug specs for the author during development.
            The default is None.

        Returns
        -------
        Qasm2AST
            The AST for the file input.

        """
        with open(filepath, "r") as _f:
            qasm_str = _f.read()
        if name is None:
            name = os.path.basename(filepath)
        return self.ast_qasm_str(qasm_str, filepath, name, dbg)

    def do_it(
        self,
        *,
        filename: str = "OpenQASM string source",
        dbg: list = None,
        profile_sortby: str = None,
    ) -> Tuple:
        """
        Create the ast and the circuit corresponding to the OpenQASM input.

        Parameters
        ----------
        filename : str, optional
            The name to assign as a "filename" for source provided as string.
            The default is None.
        dbg : list, optional
            "Magic" debug specs for the author during development.
            The default is None.
        profile_sortby : str, optional
            profile_sortby : str, optional
            If set, the operations is profiled.
            Set by providing the sort sequence for performance data.
            Sequence is one or more of the following literal strings
            separated by spaces in a single string, e.g., "calls cumtime file".
            The possible values are:
            'calls' == call count
            'cumtime' == cumulative time
            'file' == file name
            'module' == file name
            'ncalls' == call count
            'pcalls' == primitive call count
            'line' == line number
            'name' == function name
            'nfl' == name/file/line
            'stdname' == standard name
            'time' == internal time
            'tottime' == internal time
            The default is None.

        Raises
        ------
        QasmError
            If error encountered in translation.

        Returns
        -------
        Tuple
            (the quantum circuit, the original AST)

        """

        self.ast = None
        self.qt = None
        self.qc = None

        if profile_sortby:
            import cProfile

            pr = cProfile.Profile()
            pr.enable()

        if self.qasm_filepath:
            self.ast = self.ast_qasm_file(self.qasm_filepath, self.name, dbg)
        elif self.qasm_str:
            self.ast = self.ast_qasm_str(self.qasm_str, filename, self.name, dbg)
        else:
            raise QasmError("No OpenQASM source provided, neither a string nor a filename")

        self.qt = Qasm2Translator(self.ast)

        try:
            self.qc = self.qt.translate()
        except Exception as ex:  # pylint: disable=broad-except
            print(
                "Error encountered in loading:\nMessage:\n{}\nCause:\n{}\nContext:\n{}".format(
                    ex, ex.__cause__, ex.__context__
                )
            )
            traceback.print_tb(ex.__traceback__)
        if profile_sortby:
            pr.disable()
            _s = io.StringIO()
            ps = pstats.Stats(pr, stream=_s).sort_stats(profile_sortby)
            _ = ps.print_stats()
            sys.stderr.write(_s.getvalue())

        return (self.qc, self.ast)
