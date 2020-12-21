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

import numpy as np

from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterVector
from qiskit.circuit.library import RandomPauli


class TestRandomPauli(QiskitTestCase):
    """Test the Random Pauli circuit."""

    def test_random_pauli(self):
        """Test the Random Pauli circuit."""
        circuit = RandomPauli(4, seed=12, reps=1)

        qr = QuantumRegister(4, 'q')
        params = circuit.ordered_parameters
        expected = QuantumCircuit(qr)
        # initial RYs
        expected.ry(np.pi / 4, qr)

        # first random Pauli layer
        expected.ry(params[0], 0)
        expected.rx(params[1], 1)
        expected.rz(params[2], 2)
        expected.rz(params[3], 3)

        # entanglement
        expected.cz(0, 1)
        expected.cz(2, 3)
        expected.cz(1, 2)

        # second random Pauli layer
        expected.rx(params[4], 0)
        expected.rx(params[5], 1)
        expected.rx(params[6], 2)
        expected.rx(params[7], 3)

        self.assertEqual(circuit, expected)


if __name__ == '__main__':
    unittest.main()
