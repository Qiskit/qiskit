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

"""Tests utility functions for QuantumState classes."""

import unittest
import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.quantum_info.states import Statevector, DensityMatrix
from qiskit.quantum_info.states import partial_trace, shannon_entropy, schmidt_decomposition


class TestStateUtils(QiskitTestCase):
    """Test utility functions for QuantumState classes."""

    def test_statevector_partial_trace(self):
        """Test partial_trace function on statevectors"""
        psi = Statevector.from_label("10+")
        self.assertEqual(partial_trace(psi, [0, 1]), DensityMatrix.from_label("1"))
        self.assertEqual(partial_trace(psi, [0, 2]), DensityMatrix.from_label("0"))
        self.assertEqual(partial_trace(psi, [1, 2]), DensityMatrix.from_label("+"))
        self.assertEqual(partial_trace(psi, [0]), DensityMatrix.from_label("10"))
        self.assertEqual(partial_trace(psi, [1]), DensityMatrix.from_label("1+"))
        self.assertEqual(partial_trace(psi, [2]), DensityMatrix.from_label("0+"))
        self.assertEqual(partial_trace(psi, []), DensityMatrix(psi))

    def test_density_matrix_partial_trace(self):
        """Test partial_trace function on density matrices"""
        rho = DensityMatrix.from_label("10+")
        self.assertEqual(partial_trace(rho, [0, 1]), DensityMatrix.from_label("1"))
        self.assertEqual(partial_trace(rho, [0, 2]), DensityMatrix.from_label("0"))
        self.assertEqual(partial_trace(rho, [1, 2]), DensityMatrix.from_label("+"))
        self.assertEqual(partial_trace(rho, [0]), DensityMatrix.from_label("10"))
        self.assertEqual(partial_trace(rho, [1]), DensityMatrix.from_label("1+"))
        self.assertEqual(partial_trace(rho, [2]), DensityMatrix.from_label("0+"))
        self.assertEqual(partial_trace(rho, []), rho)

    def test_shannon_entropy(self):
        """Test shannon_entropy function"""
        input_pvec = np.array([0.5, 0.3, 0.07, 0.1, 0.03])
        # Base 2
        self.assertAlmostEqual(1.7736043871504037, shannon_entropy(input_pvec))
        # Base e
        self.assertAlmostEqual(1.229368880382052, shannon_entropy(input_pvec, np.e))
        # Base 10
        self.assertAlmostEqual(0.533908120973504, shannon_entropy(input_pvec, 10))

    def test_schmidt_decomposition(self):
        """Test schmidt_decomposition function"""

        # separable 2-level system without subsystem permutation
        target = Statevector.from_label("10l")
        schmidt_comps = schmidt_decomposition(Statevector.from_label("10l"), [0])
        state = Statevector(sum(suv[0] * np.kron(suv[1], suv[2]) for suv in schmidt_comps))
        self.assertEqual(state, target)

        # separable 2-level system with subsystem permutation
        target = Statevector.from_label("0l1")
        schmidt_comps = schmidt_decomposition(Statevector.from_label("l10"), [2, 1])
        state = Statevector(sum(suv[0] * np.kron(suv[1], suv[2]) for suv in schmidt_comps))
        self.assertEqual(state, target)

        # entangled 2-level system
        target = 1 / np.sqrt(2) * (Statevector.from_label("00") + Statevector.from_label("11"))
        schmidt_comps = schmidt_decomposition(target, [0])
        state = Statevector(sum(suv[0] * np.kron(suv[1], suv[2]) for suv in schmidt_comps))
        self.assertEqual(state, target)

        # entangled 3-level system
        target = Statevector(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]) * 1 / np.sqrt(3), dims=(3, 3))
        schmidt_comps = schmidt_decomposition(target, [0])
        state = Statevector(
            sum(suv[0] * np.kron(suv[1], suv[2]) for suv in schmidt_comps), dims=(3, 3)
        )
        self.assertEqual(state, target)


if __name__ == "__main__":
    unittest.main()
