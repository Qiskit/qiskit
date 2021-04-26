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
Functional interface to Qasm2 source loading and dumping
"""

from os import linesep
from typing import List, BinaryIO, TextIO
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from .qasm import Qasm
from .funhelp import qasm_load, qasm_dump


def _load_from_string(qasm_src: str or List[str]) -> QuantumCircuit:
    """

    Parameters
    ----------
    qasm_src : str or List[str]
        Qasm program source as string or list of string.

    Raises
    ------
    QiskitError
        If unknown loader.

    Returns
    -------
    QuantumCircuit
        Circuit factoried from Qasm src.

    """

    if isinstance(qasm_src, list):
        qasm_src = ''.join(s + linesep for s in qasm_src)
    qasm = Qasm(data=qasm_src)
    circ = qasm_load(qasm)
    return circ


def _load_from_file(filename: str) -> QuantumCircuit:
    """

    Parameters
    ----------
    filename : str
        Filepath to qasm program source.

    Returns
    -------
    QuantumCircuit
        Circuit factoried from Qasm src.

    """

    qasm = Qasm(filename=filename)
    return qasm_load(qasm)


def load(data: str or List[str] = None,
         filename: str = None) -> QuantumCircuit:
    """


    Parameters
    ----------
    data : str or List[str], optional
        Qasm program source as string or list of string. The default is None.
    filename : str, optional
        Filepath to qasm program source. The default is None.

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
        circ = _load_from_string(data)
    elif filename:
        circ = _load_from_file(filename)
    return circ


def dump(qc: QuantumCircuit,
         file: BinaryIO or TextIO = None,
         filename: str = None) -> str:
    """
    Decompile a QuantumCircuit into Return OpenQASM string

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to decompile ("dump")

    file : BinaryIO or TextIO, optional
        File object to write to as well as return str
        Written in UTF-8
        Caller must close file.
        Mutually exclusive with filename=
        The default is None.

    filename : str, optional
        Name of file to write dump to as well as return str
        Mutually exclusive with file=
        The default is None.

    Raises
    ------
    QiskitError
        If both filename and file

    QasmError
        If circuit has free parameters.

    Returns
    -------
    str
        OpenQASM source for circuit.

    """
    if filename and file:
        raise QiskitError("dump: file= and filename= are mutually exclusive")

    qasm_src = qasm_dump(qc)

    if filename:
        f_f = open(filename, 'w')
        f_f.write(qasm_src)
        f_f.close()
    elif file:
        if 'b' in file.mode:
            file.write(bytes(qasm_src, 'utf-8'))
        else:
            file.write(qasm_src)
    return qasm_src
