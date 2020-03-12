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
from os import linesep
from typing import List
from qiskit import QuantumCircuit, QiskitError
from qiskit.qasm import Qasm
from qiskit.converters import ast_to_dag  # pylint: disable=cyclic-import
from qiskit.converters import dag_to_circuit


def _qasm_load(qasm: Qasm) -> QuantumCircuit:
    ast = qasm.parse()
    dag = ast_to_dag(ast)
    return dag_to_circuit(dag)

def _load_from_string(qasm_src: str or List[str],
                      loader: str = 'qasm',
                      include_path: str = '') -> QuantumCircuit:
    """

    Parameters
    ----------
    qasm_src : str or List[str]
        Qasm program source as string or list of string.
    loader : str, optional
        Canonical name of desired loader. The default is 'qasm'.
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
    if loader == 'qasm':
        if isinstance(qasm_src, list):
            qasm_src = ''.join(s + linesep for s in qasm_src)
        qasm = Qasm(data=qasm_src)
        circ = _qasm_load(qasm)
    elif loader == 'nuqasm2':
        from nuqasm2 import load_string
        circ = load_string(qasm_src, include_path)
    else:
        raise QiskitError("load qasm: unknown loader")
    return circ


def _load_from_file(filename: str or List[str],
                    loader: str = 'qasm',
                    include_path: str = '') -> QuantumCircuit:
    """

    Parameters
    ----------
    filename : str
        Filepath to qasm program source.
    loader : str, optional
        Canonical name of desired loader. The default is 'qasm'.
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
    qasm = Qasm(filename=filename)
    if loader == 'qasm':
        circ = _qasm_load(qasm)
    elif loader == 'nuqasm2':
        from nuqasm2 import load_from_file
        circ = load_from_file(filename, include_path)
    else:
        raise QiskitError("load qasm: unknown loader")
    return circ

def load(data: str  or List[str] = None,
         filename: str = None,
         loader: str = 'qasm',
         include_path: str = '') -> QuantumCircuit:
    """


    Parameters
    ----------
    data : str or List[str], optional
        Qasm program source as string or list of string. The default is None.
    filename : str, optional
        Filepath to qasm program source. The default is None.
    loader : str, optional
        DESCRIPTION. The default is 'qasm'.
    include_path : str, optional
        Loader-specific include path for qasm include directives.
        The default is ''.

    Raises
    ------
    QiskitError
        If unknown loader or if both filename and data or neither filename nor data.

    Returns
    -------
    QuantumCircuit
        DESCRIPTION.

    """
    if (not data and not filename) or (data and filename):
        raise QiskitError("To load, either filename or data (and not both) must be provided.")
    circ = None
    if data:
        circ = _load_from_string(data, loader=loader, include_path=include_path)
    if filename:
        circ = _load_from_file(filename, loader=loader, include_path=include_path)
    return circ
