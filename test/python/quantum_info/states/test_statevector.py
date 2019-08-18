# -*- coding: utf-8 -*-

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

"""Tests for Statevector quantum state class."""

import unittest
import logging
import numpy as np
from numpy.testing import assert_allclose

from qiskit.test import QiskitTestCase
from qiskit import QiskitError
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.extensions.standard import HGate

from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info.states import Statevector
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.predicates import matrix_equal

logger = logging.getLogger(__name__)


class TestStatevector(QiskitTestCase):
    """Tests for Statevector class."""

    @classmethod
    def rand_vec(cls, n, normalize=False):
        """Return complex vector or statevector"""
        seed = np.random.randint(0, np.iinfo(np.int32).max)
        logger.debug("rand_vec RandomState seeded with seed=%s", seed)
        rng = np.random.RandomState(seed)

        vec = rng.rand(n) + 1j * rng.rand(n)
        if normalize:
            vec /= np.sqrt(np.dot(vec, np.conj(vec)))
        return vec

    def test_init_array_qubit(self):
        """Test subsystem initialization from N-qubit array."""
        # Test automatic inference of qubit subsystems
        vec = self.rand_vec(8)
        for dims in [None, 8]:
            state = Statevector(vec, dims=dims)
            assert_allclose(state.data, vec)
            self.assertEqual(state.dim, 8)
            self.assertEqual(state.dims(), (2, 2, 2))

    def test_init_array(self):
        """Test initialization from array."""
        vec = self.rand_vec(3)
        statevec = Statevector(vec)
        assert_allclose(statevec.data, vec)
        self.assertEqual(statevec.dim, 3)
        self.assertEqual(statevec.dims(), (3,))

        vec = self.rand_vec(2 * 3 * 4)
        state = Statevector(vec, dims=[2, 3, 4])
        assert_allclose(state.data, vec)
        self.assertEqual(state.dim, 2 * 3 * 4)
        self.assertEqual(state.dims(), (2, 3, 4))

    def test_init_array_except(self):
        """Test initialization exception from array."""
        vec = self.rand_vec(4)
        self.assertRaises(QiskitError, Statevector, vec, dims=[4, 2])
        self.assertRaises(QiskitError, Statevector, vec, dims=[2, 4])
        self.assertRaises(QiskitError, Statevector, vec, dims=5)

    def test_init_statevector(self):
        """Test initialization from Statevector."""
        vec1 = Statevector(self.rand_vec(4))
        vec2 = Statevector(vec1)
        self.assertEqual(vec1, vec2)

    def test_from_circuit(self):
        """Test initialization from a circuit."""
        # random unitaries
        u0 = random_unitary(2).data
        u1 = random_unitary(2).data
        # add to circuit
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.unitary(u0, [qr[0]])
        circ.unitary(u1, [qr[1]])
        target = Statevector(np.kron(u1, u0).dot([1, 0, 0, 0]))
        vec = Statevector.from_instruction(circ)
        self.assertEqual(vec, target)

        # Test tensor product of 1-qubit gates
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.x(1)
        circuit.ry(np.pi / 2, 2)
        target = Statevector.from_label('000').evolve(Operator(circuit))
        psi = Statevector.from_instruction(circuit)
        self.assertEqual(psi, target)

        # Test decomposition of Controlled-u1 gate
        lam = np.pi / 4
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.h(1)
        circuit.cu1(lam, 0, 1)
        target = Statevector.from_label('00').evolve(Operator(circuit))
        psi = Statevector.from_instruction(circuit)
        self.assertEqual(psi, target)

        # Test decomposition of controlled-H gate
        circuit = QuantumCircuit(2)
        circ.x(0)
        circuit.ch(0, 1)
        target = Statevector.from_label('00').evolve(Operator(circuit))
        psi = Statevector.from_instruction(circuit)
        self.assertEqual(psi, target)

    def test_from_instruction(self):
        """Test initialization from an instruction."""
        target = np.dot(HGate().to_matrix(), [1, 0])
        vec = Statevector.from_instruction(HGate()).data
        global_phase_equivalent = matrix_equal(vec, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

    def test_from_label(self):
        """Test initialization from a label"""
        x_p = Statevector(np.array([1, 1]) / np.sqrt(2))
        x_m = Statevector(np.array([1, -1]) / np.sqrt(2))
        y_p = Statevector(np.array([1, 1j]) / np.sqrt(2))
        y_m = Statevector(np.array([1, -1j]) / np.sqrt(2))
        z_p = Statevector(np.array([1, 0]))
        z_m = Statevector(np.array([0, 1]))

        label = '01'
        target = z_p.tensor(z_m)
        self.assertEqual(target, Statevector.from_label(label))

        label = '+-'
        target = x_p.tensor(x_m)
        self.assertEqual(target, Statevector.from_label(label))

        label = 'rl'
        target = y_p.tensor(y_m)
        self.assertEqual(target, Statevector.from_label(label))

    def test_equal(self):
        """Test __eq__ method"""
        for _ in range(10):
            vec = self.rand_vec(4)
            self.assertEqual(Statevector(vec),
                             Statevector(vec.tolist()))

    def test_rep(self):
        """Test Operator representation string property."""
        state = Statevector(self.rand_vec(2))
        self.assertEqual(state.rep, 'Statevector')

    def test_copy(self):
        """Test Statevector copy method"""
        for _ in range(5):
            vec = self.rand_vec(4)
            orig = Statevector(vec)
            cpy = orig.copy()
            cpy._data[0] += 1.0
            self.assertFalse(cpy == orig)

    def test_is_valid(self):
        """Test is_valid method."""
        state = Statevector([1, 1])
        self.assertFalse(state.is_valid())
        for _ in range(10):
            state = Statevector(self.rand_vec(4, normalize=True))
            self.assertTrue(state.is_valid())

    def test_to_operator(self):
        """Test to_operator method for returning projector."""
        for _ in range(10):
            vec = self.rand_vec(4)
            target = Operator(np.outer(vec, np.conj(vec)))
            op = Statevector(vec).to_operator()
            self.assertEqual(op, target)

    def test_evolve(self):
        """Test _evolve method."""
        for _ in range(10):
            op = random_unitary(4)
            vec = self.rand_vec(4)
            target = Statevector(np.dot(op.data, vec))
            evolved = Statevector(vec).evolve(op)
            self.assertEqual(target, evolved)

    def test_evolve_subsystem(self):
        """Test subsystem _evolve method."""
        # Test evolving single-qubit of 3-qubit system
        for _ in range(5):
            vec = self.rand_vec(8)
            state = Statevector(vec)
            op0 = random_unitary(2)
            op1 = random_unitary(2)
            op2 = random_unitary(2)

            # Test evolve on 1-qubit
            op = op0
            op_full = Operator(np.eye(4)).tensor(op)
            target = Statevector(np.dot(op_full.data, vec))
            self.assertEqual(state.evolve(op, qargs=[0]), target)

            # Evolve on qubit 1
            op_full = Operator(np.eye(2)).tensor(op).tensor(np.eye(2))
            target = Statevector(np.dot(op_full.data, vec))
            self.assertEqual(state.evolve(op, qargs=[1]), target)

            # Evolve on qubit 2
            op_full = op.tensor(np.eye(4))
            target = Statevector(np.dot(op_full.data, vec))
            self.assertEqual(state.evolve(op, qargs=[2]), target)

            # Test evolve on 2-qubits
            op = op1.tensor(op0)

            # Evolve on qubits [0, 2]
            op_full = op1.tensor(np.eye(2)).tensor(op0)
            target = Statevector(np.dot(op_full.data, vec))
            self.assertEqual(state.evolve(op, qargs=[0, 2]), target)

            # Evolve on qubits [2, 0]
            op_full = op0.tensor(np.eye(2)).tensor(op1)
            target = Statevector(np.dot(op_full.data, vec))
            self.assertEqual(state.evolve(op, qargs=[2, 0]), target)

            # Test evolve on 3-qubits
            op = op2.tensor(op1).tensor(op0)

            # Evolve on qubits [0, 1, 2]
            op_full = op
            target = Statevector(np.dot(op_full.data, vec))
            self.assertEqual(state.evolve(op, qargs=[0, 1, 2]), target)

            # Evolve on qubits [2, 1, 0]
            op_full = op0.tensor(op1).tensor(op2)
            target = Statevector(np.dot(op_full.data, vec))
            self.assertEqual(state.evolve(op, qargs=[2, 1, 0]), target)

    def test_conjugate(self):
        """Test conjugate method."""
        for _ in range(10):
            vec = self.rand_vec(4)
            target = Statevector(np.conj(vec))
            state = Statevector(vec).conjugate()
            self.assertEqual(state, target)

    def test_expand(self):
        """Test expand method."""
        for _ in range(10):
            vec0 = self.rand_vec(2)
            vec1 = self.rand_vec(3)
            target = np.kron(vec1, vec0)
            state = Statevector(vec0).expand(Statevector(vec1))
            self.assertEqual(state.dim, 6)
            self.assertEqual(state.dims(), (2, 3))
            assert_allclose(state.data, target)

    def test_tensor(self):
        """Test tensor method."""
        for _ in range(10):
            vec0 = self.rand_vec(2)
            vec1 = self.rand_vec(3)
            target = np.kron(vec0, vec1)
            state = Statevector(vec0).tensor(Statevector(vec1))
            self.assertEqual(state.dim, 6)
            self.assertEqual(state.dims(), (3, 2))
            assert_allclose(state.data, target)

    def test_add(self):
        """Test add method."""
        for _ in range(10):
            vec0 = self.rand_vec(4)
            vec1 = self.rand_vec(4)
            state0 = Statevector(vec0)
            state1 = Statevector(vec1)
            self.assertEqual(state0.add(state1), Statevector(vec0 + vec1))
            self.assertEqual(state0 + state1, Statevector(vec0 + vec1))

    def test_add_except(self):
        """Test add method raises exceptions."""
        state1 = Statevector(self.rand_vec(2))
        state2 = Statevector(self.rand_vec(3))
        self.assertRaises(QiskitError, state1.add, state2)

    def test_subtract(self):
        """Test subtract method."""
        for _ in range(10):
            vec0 = self.rand_vec(4)
            vec1 = self.rand_vec(4)
            state0 = Statevector(vec0)
            state1 = Statevector(vec1)
            self.assertEqual(state0.subtract(state1), Statevector(vec0 - vec1))
            self.assertEqual(state0 - state1, Statevector(vec0 - vec1))

    def test_subtract_except(self):
        """Test subtract method raises exceptions."""
        state1 = Statevector(self.rand_vec(2))
        state2 = Statevector(self.rand_vec(3))
        self.assertRaises(QiskitError, state1.subtract, state2)

    def test_multiply(self):
        """Test multiply method."""
        for _ in range(10):
            vec = self.rand_vec(4)
            state = Statevector(vec)
            val = np.random.rand() + 1j * np.random.rand()
            self.assertEqual(state.multiply(val), Statevector(val * vec))
            self.assertEqual(val * state, Statevector(val * state))

    def test_negate(self):
        """Test negate method"""
        for _ in range(10):
            vec = self.rand_vec(4)
            state = Statevector(vec)
            self.assertEqual(-state, Statevector(-1 * vec))

    def test_equiv(self):
        """Test negate method"""
        vec = np.array([1, 0, 0, -1j]) / np.sqrt(2)
        phase = np.exp(-1j * np.pi / 4)
        statevec = Statevector(vec)
        self.assertTrue(statevec.equiv(phase * vec))
        self.assertTrue(statevec.equiv(Statevector(phase * vec)))
        self.assertFalse(statevec.equiv(2 * vec))


if __name__ == '__main__':
    unittest.main()
