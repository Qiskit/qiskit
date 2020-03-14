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
Functional interface to Qasm2 source loading and unloading
Supersede QuantumCircuit member functions
Provide for pluggable qasm translator
Based on conversation with Dr. Luciano Bello
@author: jax
"""
from importlib import import_module
from os import linesep
from typing import List
from qiskit import QuantumCircuit, QiskitError
from qiskit.qasm import Qasm
from .funhelp import qasm_load, qasm_unload


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


def unload(qc: QuantumCircuit,
           unloader: str = None,
           include_path: str = None) -> str:
    """
    Decompile a QuantumCircuit into Return OpenQASM string

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to decompile ("unload")
    unloader : str. optional
        Name of module with functional attribute
            unload(qc: QuantumCircuit,
                   include_path: str = None) -> QuantumCircuit:
        ... to use for qasm translation.
        None means "use Qiskit qasm"
        The default is None.
    include_path: str, optional
        Unlaoder-specific include path for qasm include directives

    Returns
    -------
    str
        OpenQASM source for circuit.

    """
    qasm_src = None
    if not unloader:
        qasm_src = qasm_unload(qc)
    else:
        m_m = import_module(unloader)
        qasm_src = getattr(m_m, 'unload')(qc, include_path=include_path)
    return qasm_src
