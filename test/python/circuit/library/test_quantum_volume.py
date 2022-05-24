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

"""Test library of quantum volume circuits."""

import unittest

from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QuantumVolume
from qiskit.quantum_info import Operator
from qiskit.quantum_info.random import random_unitary


class TestQuantumVolumeLibrary(QiskitTestCase):
    """Test library of quantum volume quantum circuits."""

    def test_qv(self):
        """Test qv circuit."""
        circuit = QuantumVolume(2, 2, seed=2, classical_permutation=False)
        expected = QuantumCircuit(2)
        expected.swap(0, 1)
        expected.append(random_unitary(4, seed=837), [0, 1])
        expected.append(random_unitary(4, seed=262), [0, 1])
        expected = Operator(expected)
        simulated = Operator(circuit)
        self.assertTrue(expected.equiv(simulated))


if __name__ == "__main__":
    unittest.main()
