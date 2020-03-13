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
Support via qiskit.qasm for functional interface
to Qasm2 source loading and unloading in functions.py
@author: jax
"""
# from importlib import import_module
# from os import linesep
# from typing import List
from qiskit import QuantumCircuit  # , QiskitError
from qiskit.qasm import Qasm
from qiskit.converters import ast_to_dag
from qiskit.converters import dag_to_circuit


def qasm_load(qasm: Qasm) -> QuantumCircuit:
    """


    Parameters
    ----------
    qasm : Qasm
        The Qasm object of source to load.

    Returns
    -------
    QuantumCircuit
        The resulting QuantumCircuit.

    """

    ast = qasm.parse()
    dag = ast_to_dag(ast)
    return dag_to_circuit(dag)
