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
from qiskit.quantum_info import random_unitary, Operator, average_gate_fidelity
from qiskit.test import QiskitTestCase
from qiskit.circuit.library import (
    SwapGate,
    iSwapGate,
    CXGate,
    IGate,
    XGate,
    RXXGate,
    RYYGate,
    RZZGate,
)

_EPS = 1e-12


@ddt
class TestSiSwapSynth(QiskitTestCase):
    """Test synthesis of SU(4)s over SiSwap basis."""

    @combine(seed=range(50))
    def test_siswap_random(self, seed):
        """Test synthesis of 50 random SU(4)s."""
        u = random_unitary(4, seed=seed)
        decomposer = SiSwapDecomposer(euler_basis=["rz", "ry"])
        circuit = decomposer(u)
        self.assertLessEqual(circuit.count_ops().get("siswap", 0), 3)
        self.assertEqual(Operator(circuit), Operator(u))

    @combine(seed=range(50, 100), basis_fidelity=[0.98, 0.99, 1.0])
    def test_siswap_random_approx(self, seed, basis_fidelity):
        """Test synthesis of 50 random SU(4)s with approximation."""
        u = random_unitary(4, seed=seed)
        decomposer = SiSwapDecomposer(euler_basis=["rz", "ry"])
        circuit = decomposer(u, basis_fidelity=basis_fidelity)
        self.assertLessEqual(circuit.count_ops().get("siswap", 0), 3)
        self.assertGreaterEqual(average_gate_fidelity(Operator(circuit), u), basis_fidelity - _EPS)

    @combine(corner=[SwapGate(), SwapGate().power(1 / 2), SwapGate().power(1 / 32)])
    def test_siswap_corners_3_use(self, corner):
        """Test synthesis of some 3-use special corner cases."""
        u = Operator(corner)
        decomposer = SiSwapDecomposer(euler_basis=["rz", "ry"])
        circuit = decomposer(u)
        self.assertEqual(circuit.count_ops().get("siswap", 0), 3)
        self.assertEqual(Operator(circuit), Operator(u))

    @combine(
        corner=[
            iSwapGate(),
            CXGate(),
            CXGate().power(-1 / 2),
        ]
    )
    def test_siswap_corners_2_uses(self, corner):
        """Test synthesis of some 2-use special corner cases."""
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
        """Test synthesis of some 1-use special corner cases."""
        u = Operator(corner)
        decomposer = SiSwapDecomposer(euler_basis=["u"])
        circuit = decomposer(u)
        self.assertEqual(circuit.count_ops().get("siswap", 0), 1)
        self.assertEqual(Operator(circuit), Operator(u))

    @combine(corner=[Operator(IGate()) ^ Operator(IGate()), Operator(IGate()) ^ Operator(XGate())])
    def test_siswap_corners_0_use(self, corner):
        """Test synthesis of some 0-use special corner cases."""
        u = Operator(corner)
        decomposer = SiSwapDecomposer(euler_basis=["u"])
        circuit = decomposer(u)
        self.assertEqual(circuit.count_ops().get("siswap", 0), 0)
        self.assertEqual(Operator(circuit), Operator(u))

    @combine(
        p=[
            [np.pi / 4, np.pi / 6, -np.pi / 12],  # red and blue intersection (bottom face)
            [np.pi / 4, np.pi / 6, np.pi / 12],  # inverse of above (and locally equivalent)
            [np.pi / 4, np.pi / 8, np.pi / 8],  # half-way between CX and SWAP (peak red polytope)
            [np.pi / 4, np.pi / 8, -np.pi / 8],  # inverse of above (and locally equivalent)
            [np.pi / 4, np.pi / 16, np.pi / 16],  # quarter-way between CX and SWAP
            [np.pi / 4, np.pi / 16, np.pi / 16],  # inverse of above (and locally equivalent)
            [np.pi / 6, np.pi / 8, np.pi / 24],  # red and blue and green intersection
            [np.pi / 6, np.pi / 8, -np.pi / 24],  # inverse of above (not locally equivalent)
            [np.pi / 16, np.pi / 24, np.pi / 48],  # red and blue and purple intersection
            [np.pi / 16, np.pi / 24, -np.pi / 48],  # inverse of above (not locally equivalent)
            [np.pi / 6, np.pi / 12, -np.pi / 16],  # inside red polytope
        ]
    )
    def test_siswap_special_points(self, p):
        """Test special points in the Weyl chamber related to SiSwap polytopes."""
        u = (
            Operator(RXXGate(-2 * p[0]))
            & Operator(RYYGate(-2 * p[1]))
            & Operator(RZZGate(-2 * p[2]))
        )
        decomposer = SiSwapDecomposer(euler_basis=["u"])
        circuit = decomposer(u)
        self.assertEqual(circuit.count_ops().get("siswap", 0), 2)
        self.assertEqual(Operator(circuit), Operator(u))


if __name__ == "__main__":
    unittest.main()
