# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test library of permutation logic quantum circuits."""

import unittest
import numpy as np

from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import Permutation
from qiskit.quantum_info import Operator


class TestPermutationGate(QiskitTestCase):
    """Tests for the Permutation class."""

    def test_permutation(self):
        """Test permutation circuit."""
        circuit = Permutation(num_qubits=4, pattern=[1, 0, 3, 2])
        expected = QuantumCircuit(4)
        expected.swap(0, 1)
        expected.swap(2, 3)
        expected = Operator(expected)
        simulated = Operator(circuit)
        self.assertTrue(expected.equiv(simulated))

    def test_permutation_bad(self):
        """Test that [0,..,n-1] permutation is required (no -1 for last element)."""
        self.assertRaises(CircuitError, Permutation, 4, [1, 0, -1, 2])


class TestPermutationCircuit(QiskitTestCase):
    """Tests for quantum circuits containing permutations."""

    def test_append_to_circuit(self):
        """Test method for adding Permutations to quantum circuit."""
        qc = QuantumCircuit(5)
        qc.append(Permutation(3, [1, 2, 0]), [0, 1, 2])
        qc.permutation([2, 3, 0, 1], [1, 2, 3, 4])
        self.assertIsInstance(qc.data[0].operation, Permutation)
        self.assertIsInstance(qc.data[1].operation, Permutation)

    def test_inverse(self):
        """Test inverse method for circuits with permutations."""
        qc = QuantumCircuit(5)
        qc.permutation([1, 2, 3, 0], [0, 4, 2, 1])
        qci = qc.inverse()
        qci_pattern = qci.data[0].operation.pattern
        expected_pattern = [3, 0, 1, 2]

        # The inverse permutations should be defined over the same qubits but with the
        # inverse permutation pattern.
        self.assertTrue(np.array_equal(qci_pattern, expected_pattern))
        self.assertTrue(np.array_equal(qc.data[0].qubits, qci.data[0].qubits))

    def test_reverse_ops(self):
        """Test reverse_ops method for circuits with permutations."""
        qc = QuantumCircuit(5)
        qc.permutation([1, 2, 3, 0], [0, 4, 2, 1])
        qcr = qc.reverse_ops()

        # The reversed circuit should have the permutation gate with the same pattern and over the
        # same qubits.
        self.assertTrue(np.array_equal(qc.data[0].operation.pattern, qcr.data[0].operation.pattern))
        self.assertTrue(np.array_equal(qc.data[0].qubits, qcr.data[0].qubits))

    def test_conditional(self):
        """Test adding conditional permutations."""
        qc = QuantumCircuit(5, 1)
        qc.permutation([1, 2, 0], [2, 3, 4]).c_if(0, 1)
        self.assertIsNotNone(qc.data[0].operation.condition)


if __name__ == "__main__":
    unittest.main()
