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

"""Test synthesis of two-qubit unitaries into SQiSW gates."""

import unittest
from test import combine
from ddt import ddt

from qiskit.synthesis.two_qubit import SQiSWDecomposer
from qiskit.quantum_info import random_unitary, Operator
from qiskit.circuit.library import SwapGate, iSwapGate, CXGate, IGate
from qiskit.test import QiskitTestCase


@ddt
class TestSQiSWSynth(QiskitTestCase):
    """Test the Gray-Synth algorithm."""

    @combine(seed=range(50))
    def test_sqisw_random(self, seed):
        """Test synthesis of 50 random SU(4)s."""
        u = random_unitary(4, seed=seed)
        decomposer = SQiSWDecomposer(euler_basis=["rz", "ry"])
        circuit = decomposer(u)
        self.assertLessEqual(circuit.count_ops().get("sqisw", None), 3)
        self.assertEqual(Operator(circuit), Operator(u))

    @combine(corner=[SwapGate(), SwapGate().power(1 / 2), SwapGate().power(1 / 32)])
    def test_sqisw_corners_weyl(self, corner):
        """Test synthesis of some special corner cases."""
        u = Operator(corner)
        decomposer = SQiSWDecomposer(euler_basis=["rz", "ry"])
        circuit = decomposer(u)
        self.assertEqual(circuit.count_ops().get("sqisw", None), 3)
        self.assertEqual(Operator(circuit), Operator(u))

    @combine(
        corner=[
            iSwapGate(),
            iSwapGate().power(1 / 2),
            CXGate(),
            CXGate().power(-1 / 2),
            Operator(IGate()) ^ Operator(IGate()),
        ]
    )
    def test_sqisw_corners_red(self, corner):
        """Test synthesis of some special corner cases."""
        u = Operator(corner)
        decomposer = SQiSWDecomposer(euler_basis=["u"])
        circuit = decomposer(u)
        self.assertEqual(circuit.count_ops().get("sqisw", None), 2)
        self.assertEqual(Operator(circuit), Operator(u))


if __name__ == "__main__":
    unittest.main()
