# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name,missing-docstring

"""Quick program to test the quantum information states modules."""

import unittest
import numpy as np

from qiskit import execute, QuantumRegister, QuantumCircuit, BasicAer
from qiskit.quantum_info import basis_state, random_state
from qiskit.quantum_info import state_fidelity
from qiskit.quantum_info import projector
from qiskit.quantum_info import purity
from qiskit.test import QiskitTestCase


class TestStates(QiskitTestCase):
    """Tests for qi.py"""

    def test_state_fidelity(self):
        psi1 = [0.70710678118654746, 0, 0, 0.70710678118654746]
        psi2 = [0., 0.70710678118654746, 0.70710678118654746, 0.]
        rho1 = [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]]
        mix = [[0.25, 0, 0, 0], [0, 0.25, 0, 0],
               [0, 0, 0.25, 0], [0, 0, 0, 0.25]]
        self.assertAlmostEqual(state_fidelity(psi1, psi1), 1.0, places=7,
                               msg='vector-vector input')
        self.assertAlmostEqual(state_fidelity(psi1, psi2), 0.0, places=7,
                               msg='vector-vector input')
        self.assertAlmostEqual(state_fidelity(psi1, rho1), 1.0, places=7,
                               msg='vector-matrix input')
        self.assertAlmostEqual(state_fidelity(psi1, mix), 0.25, places=7,
                               msg='vector-matrix input')
        self.assertAlmostEqual(state_fidelity(psi2, rho1), 0.0, places=7,
                               msg='vector-matrix input')
        self.assertAlmostEqual(state_fidelity(psi2, mix), 0.25, places=7,
                               msg='vector-matrix input')
        self.assertAlmostEqual(state_fidelity(rho1, psi1), 1.0, places=7,
                               msg='matrix-vector input')
        self.assertAlmostEqual(state_fidelity(rho1, rho1), 1.0, places=7,
                               msg='matrix-matrix input')
        self.assertAlmostEqual(state_fidelity(mix, mix), 1.0, places=7,
                               msg='matrix-matrix input')

    def test_state_fidelity_qubit(self):
        state0 = np.array([1.+0.j, 0.+0.j])
        state1 = np.array([0.+0.j, 1.+0.j])
        self.assertEqual(state_fidelity(state0, state0), 1.0)
        self.assertEqual(state_fidelity(state1, state1), 1.0)
        self.assertEqual(state_fidelity(state0, state1), 0.0)

    def test_projector(self):
        state0 = np.array([1.+0.j, 0.+0.j])
        state1 = projector(np.array([0.+0.j, 1.+0.j]))
        self.assertEqual(state_fidelity(state0, state0), 1.0)
        self.assertEqual(state_fidelity(state1, state1), 1.0)
        self.assertEqual(state_fidelity(state0, state1), 0.0)

    def test_basis(self):
        # reference
        state = basis_state('010', 3)
        state_ideal = np.array([0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
        state_fidelity(state, state_ideal)
        self.assertEqual(state_fidelity(state, state_ideal), 1.0)

    def test_basis_state_circuit(self):
        state = state = (basis_state('001', 3)+basis_state('111', 3))/np.sqrt(2)
        q = QuantumRegister(3)
        qc = QuantumCircuit(q)
        qc.initialize(state, [q[0], q[1], q[2]])
        backend = BasicAer.get_backend('statevector_simulator')
        qc_state = execute(qc, backend).result().get_statevector(qc)
        self.assertAlmostEqual(state_fidelity(qc_state, state), 1.0, places=7)

    def test_random_state(self):
        # this test that a random state converges to 1/d
        number = 100000
        E_P0_last = 0
        for ii in range(number):
            state = basis_state(bin(3)[2:].zfill(3), 3)
            E_P0 = (E_P0_last*ii)/(ii+1)+state_fidelity(state, random_state(2**3, seed=ii))/(ii+1)
            E_P0_last = E_P0
        self.assertAlmostEqual(E_P0, 1/8, places=2)

    def test_random_state_circuit(self):
        state = random_state(2**3, seed=40)
        q = QuantumRegister(3)
        qc = QuantumCircuit(q)
        qc.initialize(state, [q[0], q[1], q[2]])
        backend = BasicAer.get_backend('statevector_simulator')
        qc_state = execute(qc, backend).result().get_statevector(qc)
        self.assertAlmostEqual(state_fidelity(qc_state, state), 1.0, places=7)

    def test_purity_list_input(self):
        rho1 = [[1, 0], [0, 0]]
        rho2 = [[0.5, 0], [0, 0.5]]
        rho3 = 0.7 * np.array(rho1) + 0.3 * np.array(rho2)
        test_pass = (purity(rho1) == 1.0 and
                     purity(rho2) == 0.5 and
                     round(purity(rho3), 10) == 0.745)
        self.assertTrue(test_pass)

    def test_purity_1d_list_input(self):
        input_state = [1, 0]
        res = purity(input_state)
        self.assertEqual(1, res)

    def test_purity_basis_state_input(self):
        state_1 = basis_state('0', 1)
        state_2 = basis_state('11', 2)
        state_3 = basis_state('010', 3)
        self.assertEqual(purity(state_1), 1.0)
        self.assertEqual(purity(state_2), 1.0)
        self.assertEqual(purity(state_3), 1.0)

    def test_purity_pure_state(self):
        state_1 = (1/np.sqrt(2))*(basis_state('0', 1) + basis_state('1', 1))
        state_2 = (1/np.sqrt(3))*(basis_state('00', 2)
                                  + basis_state('01', 2) + basis_state('11', 2))
        state_3 = 0.5*(basis_state('000', 3) + basis_state('001', 3)
                       + basis_state('010', 3) + basis_state('100', 3))
        self.assertEqual(purity(state_1), 1.0)
        self.assertEqual(purity(state_2), 1.0)
        self.assertEqual(purity(state_3), 1.0)

    def test_purity_pure_matrix_state(self):
        state_1 = (1/np.sqrt(2))*(basis_state('0', 1) + basis_state('1', 1))
        state_1 = projector(state_1)
        state_2 = (1/np.sqrt(3))*(basis_state('00', 2)
                                  + basis_state('01', 2) + basis_state('11', 2))
        state_2 = projector(state_2)
        state_3 = 0.5*(basis_state('000', 3) + basis_state('001', 3)
                       + basis_state('010', 3) + basis_state('100', 3))
        state_3 = projector(state_3)
        self.assertAlmostEqual(purity(state_1), 1.0, places=10)
        self.assertAlmostEqual(purity(state_2), 1.0, places=10)
        self.assertEqual(purity(state_3), 1.0)

    def test_purity_mixed_state(self):
        state_1 = 0.5*(projector(basis_state('0', 1))
                       + projector(basis_state('1', 1)))
        state_2 = (1/3.0)*(projector(basis_state('00', 2))
                           + projector(basis_state('01', 2))
                           + projector(basis_state('10', 2)))
        self.assertEqual(purity(state_1), 0.5)
        self.assertEqual(purity(state_2), 1.0/3)


if __name__ == '__main__':
    unittest.main()
