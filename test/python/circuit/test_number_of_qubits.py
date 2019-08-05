# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test registerless QuantumCircuit and Gates on wires"""


from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.test import QiskitTestCase


class TestNumberOfQubits(QiskitTestCase):
    """Test number of qubits in Quantum Circuit"""
    def qubitless_circuit(self):
        """
        Check output in absence of qubits
        """
        q_reg = QuantumRegister(0)
        c_reg = ClassicalRegister(3)
        circ = QuantumCircuit(q_reg, c_reg)
        self.assertEqual(circ.n_qubits, 0)

    def qubitfull_circuit(self):
        """
        Check output in presence of qubits
        """
        q_reg = QuantumRegister(4)
        c_reg = ClassicalRegister(3)
        circ = QuantumCircuit(q_reg, c_reg)
        self.assertEqual(circ.n_qubits, 4)

    def registerless_circuit(self):
        """
        Check output for circuits with direct argument for qubits
        """
        circ = QuantumCircuit(5)
        self.assertEqual(circ.n_qubits, 5)
