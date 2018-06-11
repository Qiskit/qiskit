# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Utilities regarding the creation of QuantumCircuits from a variety of different
file formats.
"""

import os
from qiskit import QISKitError
from qiskit.qasm import Qasm
from qiskit.unroll import Unroller, CircuitBackend


def circuit_from_qasm_string(qasm_string, name=None,
                             basis_gates="id,u0,u1,u2,u3,x,y,z,h,s,sdg,t,tdg,"
                             "rx,ry,rz,cx,cy,cz,ch,crz,cu1,cu3,swap,ccx,"
                             "cswap"):

    """Construct a quantum circuit from a qasm representation (string).

    Args:
        qasm_string (str): a string of qasm, or a filename containing qasm.
        basis_gates (str): basis gates for the quantum circuit.
        name (str or None): the name of the quantum circuit after loading
            qasm text into it. If no name given, assign automatically.
    Returns:
        QuantumCircuit: circuit constructed from qasm.
    Raises:
        QISKitError: if the string is not valid QASM
    """

    node_circuit = Qasm(data=qasm_string).parse()
    unrolled_circuit = Unroller(
        node_circuit, CircuitBackend(basis_gates.split(",")))
    circuit_unrolled = unrolled_circuit.execute()
    if name:
        circuit_unrolled.name = name
    return circuit_unrolled


def circuit_from_qasm_file(qasm_file, name=None,
                           basis_gates="id,u0,u1,u2,u3,x,y,z,h,s,sdg,t,tdg,rx,"
                           "ry,rz,cx,cy,cz,ch,crz,cu1,cu3,swap,ccx,"
                           "cswap"):

    """Construct a quantum circuit from a qasm representation (file).

    Args:
        qasm_file (str): a string for the filename including its location.
        name (str or None): the name of the quantum circuit after
            loading qasm text into it. If no name is give the name is of
            the text file.
        basis_gates (str): basis gates for the quantum circuit.
    Returns:
        QuantumCircuit: circuit constructed from qasm.
    Raises:
        QISKitError: if the file cannot be read.
    """
    if not os.path.exists(qasm_file):
        raise QISKitError('qasm file "{0}" not found'.format(qasm_file))
    if not name:
        name = os.path.splitext(os.path.basename(qasm_file))[0]

    with open(qasm_file) as file:
        qasm_data = file.read()

    return circuit_from_qasm_string(
        qasm_data, name=name, basis_gates=basis_gates)
