# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring

"""Quick program to test the quantum information states modules."""

import unittest
import numpy as np
from qiskit import execute, QuantumRegister, QuantumCircuit, Aer

from qiskit.quantum_info import basis_state, random_state
from qiskit.quantum_info import state_fidelity
from qiskit.quantum_info import projector

from ..common import QiskitTestCase


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
        backend = Aer.get_backend('statevector_simulator')
        qc_state = execute(qc, backend).result().get_statevector(qc)
        self.assertAlmostEqual(state_fidelity(qc_state, state), 1.0, places=7)

    def test_random_state(self):
        # this test that a random state converges to 1/d
        number = 100000
        E_P0_last = 0
        for ii in range(number):
            state = basis_state(bin(np.random.randint(0, 8))[2:].zfill(3), 3)
            E_P0 = (E_P0_last*ii)/(ii+1)+state_fidelity(state,
                                                        random_state(3))/(ii+1)
            E_P0_last = E_P0
        self.assertAlmostEqual(E_P0, 1/8, places=2)

    def test_random_state_circuit(self):
        state = random_state(3)
        q = QuantumRegister(3)
        qc = QuantumCircuit(q)
        qc.initialize(state, [q[0], q[1], q[2]])
        backend = Aer.get_backend('statevector_simulator')
        qc_state = execute(qc, backend).result().get_statevector(qc)
        self.assertAlmostEqual(state_fidelity(qc_state, state), 1.0, places=7)


if __name__ == '__main__':
    unittest.main()
