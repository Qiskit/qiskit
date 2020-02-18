# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test library of quantum circuits."""

from qiskit.test import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import Permutation, Shift, InnerProduct


class TestBooleanLogicLibrary(QiskitTestCase):
    """Test library of boolean logic quantum circuits."""

    def test_permutation(self):
        """Test permutation circuit."""
        circuit = Permutation(n_qubits=4, pattern=[1, 0, 3, 2])
        expected = QuantumCircuit(4)
        expected.swap(0, 1)
        expected.swap(2, 3)
        self.assertEqual(circuit, expected)

    def test_shift(self):
        """Test shift circuit."""
        circuit = Shift(n_qubits=3, amount=4)
        expected = QuantumCircuit(3)
        expected.x(2)
        self.assertEqual(circuit, expected)

    def test_inner_product(self):
        """Test inner product circuit."""
        circuit = InnerProduct(n_qubits=3)
        expected = QuantumCircuit(*circuit.qregs)
        expected.cz(0, 3)
        expected.cz(1, 4)
        expected.cz(2, 5)
        self.assertEqual(circuit, expected)
