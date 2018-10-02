# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Example showing how to draw a quantum circuit using Qiskit Terra.

"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.tools.visualization import circuit_drawer


def build_bell_circuit():
    """Returns a circuit putting 2 qubits in the Bell state."""
    q = QuantumRegister(2)
    c = ClassicalRegister(2)
    qc = QuantumCircuit(q, c)
    qc.h(q[0])
    qc.cx(q[0], q[1])
    qc.measure(q, c)
    return qc

# Create the circuit
bell_circuit = build_bell_circuit()

# Provide a name to write the diagram to the filesystem
circuit_drawer(bell_circuit, filename='./bell_circuit.png')

# Use the return value with show() to display the diagram
diagram = circuit_drawer(bell_circuit)
diagram.show()
