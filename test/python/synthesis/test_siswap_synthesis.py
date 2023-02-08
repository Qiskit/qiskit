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

# pylint: disable=invalid-name

"""Test synthesis of two-qubit unitaries into SiSwap gates."""

import unittest
from test import combine
from ddt import ddt
import numpy as np

from qiskit.synthesis.su4 import SiSwapDecomposer
from qiskit.quantum_info import random_unitary, Operator
from qiskit.test import QiskitTestCase
from qiskit.circuit.library import SwapGate, iSwapGate, CXGate, IGate, SGate, XGate, RXXGate, RYYGate, RZZGate


@ddt
class TestSiSwapSynth(QiskitTestCase):
    """Test the Gray-Synth algorithm."""

    @combine(seed=range(50))
    def test_siswap_random(self, seed):
        """Test synthesis of 50 random SU(4)s."""
        u = random_unitary(4, seed=seed)
        decomposer = SiSwapDecomposer(euler_basis=["rz", "ry"])
        circuit = decomposer(u)
        self.assertLessEqual(circuit.count_ops().get("siswap", 0), 3)
        self.assertEqual(Operator(circuit), Operator(u))

    @combine(corner=[SwapGate(), SwapGate().power(1 / 2), SwapGate().power(1 / 32)])
    def test_siswap_corners_weyl(self, corner):
        """Test synthesis of some special corner cases."""
        u = Operator(corner)
        decomposer = SiSwapDecomposer(euler_basis=["rz", "ry"])
        circuit = decomposer(u)
        self.assertEqual(circuit.count_ops().get("siswap", 0), 3)
        self.assertEqual(Operator(circuit), Operator(u))

    @combine(corner=[iSwapGate(), CXGate(), CXGate().power(-1 / 2),
                     Operator(RXXGate(-2*np.pi/4)) @ Operator(RYYGate(-2*np.pi/8)) @ Operator(RZZGate(2*np.pi/8))]
             )
    def test_siswap_corners_2_uses(self, corner):
        """Test synthesis of some special corner cases."""
        u = Operator(corner)
        decomposer = SiSwapDecomposer(euler_basis=["u"])
        circuit = decomposer(u)
        self.assertEqual(circuit.count_ops().get("siswap", 0), 2)
        self.assertEqual(Operator(circuit), Operator(u))

    @combine(
        corner=[
            iSwapGate().power(1 / 2),
            iSwapGate().power(-1 / 2),
        ]
    )
    def test_siswap_corners_1_use(self, corner):
        """Test synthesis of some special corner cases."""
        u = Operator(corner)
        decomposer = SiSwapDecomposer(euler_basis=["u"])
        circuit = decomposer(u)
        self.assertEqual(circuit.count_ops().get("siswap", 0), 1)
        self.assertEqual(Operator(circuit), Operator(u))

    @combine(corner=[Operator(IGate()) ^ Operator(IGate()), Operator(IGate()) ^ Operator(XGate())])
    def test_siswap_corners_0_use(self, corner):
        """Test synthesis of some special corner cases."""
        u = Operator(corner)
        decomposer = SiSwapDecomposer(euler_basis=["u"])
        circuit = decomposer(u)
        self.assertEqual(circuit.count_ops().get("siswap", 0), 0)
        self.assertEqual(Operator(circuit), Operator(u))


if __name__ == "__main__":
    unittest.main()
