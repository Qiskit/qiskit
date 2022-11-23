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

from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import Permutation
from qiskit.quantum_info import Operator


class TestPermutationLibrary(QiskitTestCase):
    """Test library of permutation logic quantum circuits."""

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


if __name__ == "__main__":
    unittest.main()
