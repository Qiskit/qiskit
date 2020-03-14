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
    Factory OpenQASM src into QuantumCircuit
    using qiskit.qasm code.

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


def qasm_unload(qc: QuantumCircuit) -> str:
    """
    Return OpenQASM string using qiskit.qasm code.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to be disassembled into OpenQASM source

    Returns
    -------
    str
        OpenQASM source for the circuit.

    """

    string_temp = qc.header + "\n"
    string_temp += qc.extension_lib + "\n"
    for register in qc.qregs:
        string_temp += register.qasm() + "\n"
    for register in qc.cregs:
        string_temp += register.qasm() + "\n"
    unitary_gates = []
    for instruction, qargs, cargs in qc._data:
        if instruction.name == 'measure':
            qubit = qargs[0]
            clbit = cargs[0]
            string_temp += "%s %s[%d] -> %s[%d];\n" % (instruction.qasm(),
                                                       qubit.register.name, qubit.index,
                                                       clbit.register.name, clbit.index)
        else:
            string_temp += "%s %s;\n" % (instruction.qasm(),
                                         ",".join(["%s[%d]" % (j.register.name, j.index)
                                                   for j in qargs + cargs]))
        if instruction.name == 'unitary':
            unitary_gates.append(instruction)

    # this resets them, so if another call to qasm() is made the gate def is added again
    for gate in unitary_gates:
        gate._qasm_def_written = False
    return string_temp
