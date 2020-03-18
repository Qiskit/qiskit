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

# pylint: disable=invalid-name

"""Quick program to test the quantum information states modules."""

import unittest
import numpy as np

from qiskit import execute, QuantumRegister, QuantumCircuit, BasicAer
from qiskit.quantum_info import basis_state, random_state
from qiskit.quantum_info import state_fidelity
from qiskit.quantum_info import projector
from qiskit.quantum_info.states import DensityMatrix, Statevector
from qiskit.test import QiskitTestCase


class TestStates(QiskitTestCase):
    """Tests for qi.py"""

    def test_projector(self):
        """TO BE REMOVED with qiskit.quantum_info.basis_state"""
        state0 = np.array([1.+0.j, 0.+0.j])
        with self.assertWarns(DeprecationWarning):
            state1 = projector(np.array([0.+0.j, 1.+0.j]))
        self.assertEqual(state_fidelity(state0, state0), 1.0)
        self.assertEqual(state_fidelity(state1, state1), 1.0)
        self.assertEqual(state_fidelity(state0, state1), 0.0)

    def test_basis(self):
        """TO BE REMOVED with qiskit.quantum_info.basis_state"""
        with self.assertWarns(DeprecationWarning):
            state = basis_state('010', 3)
        state_ideal = np.array([0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
        state_fidelity(state, state_ideal)
        self.assertEqual(state_fidelity(state, state_ideal), 1.0)

    def test_basis_state_circuit(self):
        """TO BE REMOVED with qiskit.quantum_info.basis_state"""
        with self.assertWarns(DeprecationWarning):
            state = (basis_state('001', 3) + basis_state('111', 3))/np.sqrt(2)
        q = QuantumRegister(3)
        qc = QuantumCircuit(q)
        qc.initialize(state, [q[0], q[1], q[2]])
        backend = BasicAer.get_backend('statevector_simulator')
        qc_state = execute(qc, backend).result().get_statevector(qc)
        self.assertAlmostEqual(state_fidelity(qc_state, state), 1.0, places=7)

    def test_random_state(self):
        """TO BE REMOVED with qiskit.quantum_info.basis_state"""
        # this test that a random state converges to 1/d
        number = 1000
        E_P0_last = 0
        for ii in range(number):
            with self.assertWarns(DeprecationWarning):
                state = basis_state(bin(3)[2:].zfill(3), 3)
            E_P0 = (E_P0_last*ii)/(ii+1)+state_fidelity(state, random_state(2**3, seed=ii))/(ii+1)
            E_P0_last = E_P0
        self.assertAlmostEqual(E_P0, 1/8, places=2)

    def test_random_state_circuit(self):
        """Run initizalized circuit"""
        state = random_state(2**3, seed=40)
        q = QuantumRegister(3)
        qc = QuantumCircuit(q)
        qc.initialize(state, [q[0], q[1], q[2]])
        backend = BasicAer.get_backend('statevector_simulator')
        qc_state = execute(qc, backend).result().get_statevector(qc)
        self.assertAlmostEqual(state_fidelity(qc_state, state), 1.0, places=7)

    def statevector_to_counts(self):
        """Statevector to counts dict"""
        state = [0.70711, 0, 0, .70711]
        ans = Statevector(state).to_counts()
        self.assertAlmostEqual(ans['00'], 0.5)
        self.assertAlmostEqual(ans['11'], 0.5)

    def densitymatrix_to_counts(self):
        """DensityMatrix to counts dict"""
        state = [0.70711, 0, 0, .70711]
        ans = DensityMatrix(state).to_counts()
        self.assertAlmostEqual(ans['00'], 0.5)
        self.assertAlmostEqual(ans['11'], 0.5)


if __name__ == '__main__':
    unittest.main()
