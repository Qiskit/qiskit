# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Created on Wed Mar 11 18:03:12 2020
Functional interface to Qasm2 source loading and exporting
Supersede QuantumCircuit member functions
Provide for pluggable qasm translator
Based on conversation with Dr. Luciano Bello
@author: jax
"""
from importlib import import_module
from os import linesep
from typing import List, BinaryIO, TextIO
from qiskit import QuantumCircuit, QiskitError
from qiskit.qasm import Qasm
from .funhelp import qasm_load, qasm_export


def _load_from_string(qasm_src: str or List[str],
                      loader: str = None,
                      include_path: str = '') -> QuantumCircuit:
    """

    Parameters
    ----------
    qasm_src : str or List[str]
        Qasm program source as string or list of string.
    loader : str, optional
        Name of module with functional attribute
            load(filename: str = None,
                 data: str = None,
                 include_path: str = None) -> QuantumCircuit:
        ... to use for qasm translation.
        None means "use Qiskit qasm"
        The default is None.
    include_path : str, optional
        Loader-specific include path for qasm include directives.
        The default is ''.

    Raises
    ------
    QiskitError
        If unknown loader.

    Returns
    -------
    QuantumCircuit
        Circuit factoried from Qasm src.

    """

    circ = None
    if not loader:
        if isinstance(qasm_src, list):
            qasm_src = ''.join(s + linesep for s in qasm_src)
        qasm = Qasm(data=qasm_src)
        circ = qasm_load(qasm)
    else:
        m_m = import_module(loader)
        circ = getattr(m_m, 'load')(data=qasm_src,
                                    include_path=include_path)
    return circ


def _load_from_file(filename: str,
                    loader: str = None,
                    include_path: str = '') -> QuantumCircuit:
    """

    Parameters
    ----------
    filename : str
        Filepath to qasm program source.
    loader : str, optional
        Name of module with functional attribute
            load(filename: str = None,
                 data: str = None,
                 include_path: str = None) -> QuantumCircuit:
        ... to use for qasm translation.
        None means "use Qiskit qasm"
        The default is None.
    include_path : str, optional
        Loader-specific include path for qasm include directives.
        The default is ''.

    Returns
    -------
    QuantumCircuit
        Circuit factoried from Qasm src.

    """

    circ = None
    if not loader:
        qasm = Qasm(filename=filename)
        circ = qasm_load(qasm)
    else:
        m_m = import_module(loader)
        circ = getattr(m_m, 'load')(filename=filename,
                                    include_path=include_path)
    return circ


def load(data: str or List[str] = None,
         filename: str = None,
         loader: str = None,
         include_path: str = None) -> QuantumCircuit:
    """


    Parameters
    ----------
    data : str or List[str], optional
        Qasm program source as string or list of string. The default is None.
    filename : str, optional
        Filepath to qasm program source. The default is None.
    loader : str, optional
        Name of module with functional attribute
            load(filename: str = None,
                 data: str = None,
                 include_path: str = None) -> QuantumCircuit:
        ... to use for qasm translation.
        None means "use Qiskit qasm"
        The default is None.
    include_path : str, optional
        Loader-specific include path for qasm include directives.
        The default is None.

    Raises
    ------
    QiskitError
        If both filename and data or neither filename nor data.

    Returns
    -------
    QuantumCircuit
        The factoried circuit.

    """

    if (not data and not filename) or (data and filename):
        raise QiskitError("To load, either filename or data (and not both) must be provided.")

    circ = None

    if data:
        circ = _load_from_string(data, loader=loader, include_path=include_path)
    elif filename:
        circ = _load_from_file(filename, loader=loader, include_path=include_path)
    return circ


def export(qc: QuantumCircuit,
           exporter: str = None,
           file: BinaryIO or TextIO = None,
           filename: str = None,
           include_path: str = None,) -> str:
    """
    Decompile a QuantumCircuit into Return OpenQASM string

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to decompile ("export")
    exporter : str, optional
        Name of module with functional attribute
            export(qc: QuantumCircuit,
                   include_path: str = None) -> QuantumCircuit:
        ... to use for qasm translation.
        None means "use Qiskit qasm"
        The default is None.
    file : BinaryIO or TextIO, optional
        File object to write to as well as return str
        Caller must close file.
        Mutually exclusive with filename=
        The default is None.
    filename : str, optional
        Name of file to write export to as well as return str
        Mutually exclusive with file=
        The default is None.
    include_path: str, optional
        Unloader-specific include path for qasm include directives

    Raises
    ------
    QiskitError
        If both filename and file

    Returns
    -------
    str
        OpenQASM source for circuit.

    """
    if filename and file:
        raise QiskitError("export: file= and filename= are mutually exclusive")

    qasm_src = None

    if not exporter:
        qasm_src = qasm_export(qc)
    else:
        m_m = import_module(exporter)
        qasm_src = getattr(m_m, 'export')(qc, include_path=include_path)
    if filename:
        f_f = open(filename, 'w')
        if f_f.isinstance(BinaryIO):
            qasm_src = qasm_src.bytes()
        f_f.write(qasm_src)
        f_f.close()
    elif file:
        file.write(qasm_src)
    return qasm_src
