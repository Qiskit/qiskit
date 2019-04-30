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
Example showing how to draw a quantum circuit using Qiskit Terra.

"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


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

# Use the internal .draw() to print the circuit
print(bell_circuit)
