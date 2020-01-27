# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-docstring

"""Quick program to test the quantum information states modules."""

import unittest
import numpy as np

from qiskit import QiskitError
from qiskit.test import QiskitTestCase
from qiskit.quantum_info.states import DensityMatrix, Statevector
from qiskit.quantum_info import state_fidelity
from qiskit.quantum_info import purity


class TestStateMeasures(QiskitTestCase):
    """Tests state measure functions"""

    def test_state_fidelity_statevector(self):
        """Test state_fidelity function for statevector inputs"""

        psi1 = [0.70710678118654746, 0, 0, 0.70710678118654746]
        psi2 = [0., 0.70710678118654746, 0.70710678118654746, 0.]
        self.assertAlmostEqual(state_fidelity(psi1, psi1), 1.0, places=7,
                               msg='vector-vector input')
        self.assertAlmostEqual(state_fidelity(psi2, psi2), 1.0, places=7,
                               msg='vector-vector input')
        self.assertAlmostEqual(state_fidelity(psi1, psi2), 0.0, places=7,
                               msg='vector-vector input')

        psi1 = Statevector([0.70710678118654746, 0, 0, 0.70710678118654746])
        psi2 = Statevector([0., 0.70710678118654746, 0.70710678118654746, 0.])
        self.assertAlmostEqual(state_fidelity(psi1, psi1), 1.0, places=7,
                               msg='vector-vector input')
        self.assertAlmostEqual(state_fidelity(psi2, psi2), 1.0, places=7,
                               msg='vector-vector input')
        self.assertAlmostEqual(state_fidelity(psi1, psi2), 0.0, places=7,
                               msg='vector-vector input')

        psi1 = Statevector([1, 0, 0, 1])  # invalid state
        psi2 = Statevector([1, 0, 0, 0])
        self.assertRaises(QiskitError, state_fidelity, psi1, psi2)
        self.assertRaises(QiskitError, state_fidelity, psi1, psi2, validate=True)
        self.assertEqual(state_fidelity(psi1, psi2, validate=False), 1)

    def test_state_fidelity_density_matrix(self):
        """Test state_fidelity function for density matrix inputs"""
        rho1 = [[0.5, 0, 0, 0.5],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0.5, 0, 0, 0.5]]
        mix = [[0.25, 0, 0, 0],
               [0, 0.25, 0, 0],
               [0, 0, 0.25, 0],
               [0, 0, 0, 0.25]]
        self.assertAlmostEqual(state_fidelity(rho1, rho1), 1.0, places=7,
                               msg='matrix-matrix input')
        self.assertAlmostEqual(state_fidelity(mix, mix), 1.0, places=7,
                               msg='matrix-matrix input')
        self.assertAlmostEqual(state_fidelity(rho1, mix), 0.25, places=7,
                               msg='matrix-matrix input')

        rho1 = DensityMatrix(rho1)
        mix = DensityMatrix(mix)
        self.assertAlmostEqual(state_fidelity(rho1, rho1), 1.0, places=7,
                               msg='matrix-matrix input')
        self.assertAlmostEqual(state_fidelity(mix, mix), 1.0, places=7,
                               msg='matrix-matrix input')
        self.assertAlmostEqual(state_fidelity(rho1, mix), 0.25, places=7,
                               msg='matrix-matrix input')

        rho1 = DensityMatrix([1, 0, 0, 0])
        mix = DensityMatrix(np.diag([1, 0, 0, 1]))
        self.assertRaises(QiskitError, state_fidelity, rho1, mix)
        self.assertRaises(QiskitError, state_fidelity, rho1, mix, validate=True)
        self.assertEqual(state_fidelity(rho1, mix, validate=False), 1)

    def test_state_fidelity_mixed(self):
        """Test state_fidelity function for statevector and density matrix inputs"""
        psi1 = Statevector([0.70710678118654746, 0, 0, 0.70710678118654746])
        rho1 = DensityMatrix([[0.5, 0, 0, 0.5],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0.5, 0, 0, 0.5]])
        mix = DensityMatrix([[0.25, 0, 0, 0],
                             [0, 0.25, 0, 0],
                             [0, 0, 0.25, 0],
                             [0, 0, 0, 0.25]])
        self.assertAlmostEqual(state_fidelity(psi1, rho1), 1.0, places=7,
                               msg='vector-matrix input')
        self.assertAlmostEqual(state_fidelity(psi1, mix), 0.25, places=7,
                               msg='vector-matrix input')
        self.assertAlmostEqual(state_fidelity(rho1, psi1), 1.0, places=7,
                               msg='matrix-vector input')

    def test_purity_statevector(self):
        """Test purity function on statevector inputs"""
        psi = Statevector([1, 0, 0, 0])
        self.assertEqual(purity(psi), 1)
        self.assertEqual(purity(psi, validate=True), 1)
        self.assertEqual(purity(psi, validate=False), 1)

        psi = [0.70710678118654746, 0.70710678118654746]
        self.assertAlmostEqual(purity(psi), 1)
        self.assertAlmostEqual(purity(psi, validate=True), 1)
        self.assertAlmostEqual(purity(psi, validate=False), 1)

        psi = np.array([0.5, 0.5j, -0.5j, -0.5])
        self.assertAlmostEqual(purity(psi), 1)
        self.assertAlmostEqual(purity(psi, validate=True), 1)
        self.assertAlmostEqual(purity(psi, validate=False), 1)

        psi = Statevector([1, 0, 0, 1])
        self.assertRaises(QiskitError, purity, psi)
        self.assertRaises(QiskitError, purity, psi, validate=True)
        self.assertEqual(purity(psi, validate=False), 4)

    def test_purity_density_matrix(self):
        """Test purity function on density matrix inputs"""
        rho = DensityMatrix(np.diag([1, 0, 0, 0]))
        self.assertEqual(purity(rho), 1)
        self.assertEqual(purity(rho, validate=True), 1)
        self.assertEqual(purity(rho, validate=False), 1)

        rho = np.diag([0.25, 0.25, 0.25, 0.25])
        self.assertEqual(purity(rho), 0.25)
        self.assertEqual(purity(rho, validate=True), 0.25)
        self.assertEqual(purity(rho, validate=False), 0.25)

        rho = [[0.5, 0, 0, 0.5],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0.5, 0, 0, 0.5]]
        self.assertEqual(purity(rho), 1)
        self.assertEqual(purity(rho, validate=True), 1)
        self.assertEqual(purity(rho, validate=False), 1)

        rho = np.diag([1, 0, 0, 1])
        self.assertRaises(QiskitError, purity, rho)
        self.assertRaises(QiskitError, purity, rho, validate=True)
        self.assertEqual(purity(rho, validate=False), 2)

    def test_purity_statevector_density_matrix(self):
        """Test purity is same for equivalent statevector and density matrix inputs"""
        psi = Statevector([0.5, -0.5, 0.5j, -0.5j])
        rho = DensityMatrix(psi)
        self.assertAlmostEqual(purity(psi), purity(rho))

        psi = Statevector([0.5, 0, 0, -0.5j])
        rho = DensityMatrix(psi)
        self.assertAlmostEqual(purity(psi, validate=False), purity(rho, validate=False))

        psi = Statevector([1, 1])
        rho = DensityMatrix(psi)
        self.assertAlmostEqual(purity(psi, validate=False), purity(rho, validate=False))


if __name__ == '__main__':
    unittest.main()
