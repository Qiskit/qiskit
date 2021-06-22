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

"""Tests for Operator matrix linear operator class."""

import unittest
import logging
import copy
import numpy as np
from numpy.testing import assert_allclose
import scipy.linalg as la

from qiskit import QiskitError
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import HGate, CHGate, CXGate, QFT
from qiskit.test import QiskitTestCase
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.predicates import matrix_equal

logger = logging.getLogger(__name__)


class OperatorTestCase(QiskitTestCase):
    """Test utils for Operator"""

    # Pauli-matrix unitaries
    UI = np.eye(2)
    UX = np.array([[0, 1], [1, 0]])
    UY = np.array([[0, -1j], [1j, 0]])
    UZ = np.diag([1, -1])
    UH = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    @classmethod
    def rand_rho(cls, n):
        """Return random density matrix"""
        seed = np.random.randint(0, np.iinfo(np.int32).max)
        logger.debug("rand_rho default_rng seeded with seed=%s", seed)
        rng = np.random.default_rng(seed)

        psi = rng.random(n) + 1j * rng.random(n)
        rho = np.outer(psi, psi.conj())
        rho /= np.trace(rho)
        return rho

    @classmethod
    def rand_matrix(cls, rows, cols=None, real=False):
        """Return a random matrix."""
        seed = np.random.randint(0, np.iinfo(np.int32).max)
        logger.debug("rand_matrix default_rng seeded with seed=%s", seed)
        rng = np.random.default_rng(seed)

        if cols is None:
            cols = rows
        if real:
            return rng.random(size=(rows, cols))
        return rng.random(size=(rows, cols)) + 1j * rng.random(size=(rows, cols))

    def simple_circuit_no_measure(self):
        """Return a unitary circuit and the corresponding unitary array."""
        qr = QuantumRegister(3)
        circ = QuantumCircuit(qr)
        circ.h(qr[0])
        circ.x(qr[1])
        circ.ry(np.pi / 2, qr[2])
        y90 = (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]])
        target = Operator(np.kron(y90, np.kron(self.UX, self.UH)))
        return circ, target

    def simple_circuit_with_measure(self):
        """Return a unitary circuit with measurement."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        circ = QuantumCircuit(qr, cr)
        circ.h(qr[0])
        circ.x(qr[1])
        circ.measure(qr, cr)
        return circ


class TestOperator(OperatorTestCase):
    """Tests for Operator linear operator class."""

    def test_init_array_qubit(self):
        """Test subsystem initialization from N-qubit array."""
        # Test automatic inference of qubit subsystems
        mat = self.rand_matrix(8, 8)
        op = Operator(mat)
        assert_allclose(op.data, mat)
        self.assertEqual(op.dim, (8, 8))
        self.assertEqual(op.input_dims(), (2, 2, 2))
        self.assertEqual(op.output_dims(), (2, 2, 2))
        self.assertEqual(op.num_qubits, 3)

        op = Operator(mat, input_dims=8, output_dims=8)
        assert_allclose(op.data, mat)
        self.assertEqual(op.dim, (8, 8))
        self.assertEqual(op.input_dims(), (2, 2, 2))
        self.assertEqual(op.output_dims(), (2, 2, 2))
        self.assertEqual(op.num_qubits, 3)

    def test_init_array(self):
        """Test initialization from array."""
        mat = np.eye(3)
        op = Operator(mat)
        assert_allclose(op.data, mat)
        self.assertEqual(op.dim, (3, 3))
        self.assertEqual(op.input_dims(), (3,))
        self.assertEqual(op.output_dims(), (3,))
        self.assertIsNone(op.num_qubits)

        mat = self.rand_matrix(2 * 3 * 4, 4 * 5)
        op = Operator(mat, input_dims=[4, 5], output_dims=[2, 3, 4])
        assert_allclose(op.data, mat)
        self.assertEqual(op.dim, (4 * 5, 2 * 3 * 4))
        self.assertEqual(op.input_dims(), (4, 5))
        self.assertEqual(op.output_dims(), (2, 3, 4))
        self.assertIsNone(op.num_qubits)

    def test_init_array_except(self):
        """Test initialization exception from array."""
        mat = self.rand_matrix(4, 4)
        self.assertRaises(QiskitError, Operator, mat, input_dims=[4, 2])
        self.assertRaises(QiskitError, Operator, mat, input_dims=[2, 4])
        self.assertRaises(QiskitError, Operator, mat, input_dims=5)

    def test_init_operator(self):
        """Test initialization from Operator."""
        op1 = Operator(self.rand_matrix(4, 4))
        op2 = Operator(op1)
        self.assertEqual(op1, op2)

    def test_circuit_init(self):
        """Test initialization from a circuit."""
        # Test tensor product of 1-qubit gates
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.x(1)
        circuit.ry(np.pi / 2, 2)
        op = Operator(circuit)
        y90 = (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]])
        target = np.kron(y90, np.kron(self.UX, self.UH))
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

        # Test decomposition of Controlled-Phase gate
        lam = np.pi / 4
        circuit = QuantumCircuit(2)
        circuit.cp(lam, 0, 1)
        op = Operator(circuit)
        target = np.diag([1, 1, 1, np.exp(1j * lam)])
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

        # Test decomposition of controlled-H gate
        circuit = QuantumCircuit(2)
        circuit.ch(0, 1)
        op = Operator(circuit)
        target = np.kron(self.UI, np.diag([1, 0])) + np.kron(self.UH, np.diag([0, 1]))
        global_phase_equivalent = matrix_equal(op.data, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

    def test_instruction_init(self):
        """Test initialization from a circuit."""
        gate = CXGate()
        op = Operator(gate).data
        target = gate.to_matrix()
        global_phase_equivalent = matrix_equal(op, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

        gate = CHGate()
        op = Operator(gate).data
        had = HGate().to_matrix()
        target = np.kron(had, np.diag([0, 1])) + np.kron(np.eye(2), np.diag([1, 0]))
        global_phase_equivalent = matrix_equal(op, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

    def test_circuit_init_except(self):
        """Test initialization from circuit with measure raises exception."""
        circuit = self.simple_circuit_with_measure()
        self.assertRaises(QiskitError, Operator, circuit)

    def test_equal(self):
        """Test __eq__ method"""
        mat = self.rand_matrix(2, 2, real=True)
        self.assertEqual(Operator(np.array(mat, dtype=complex)), Operator(mat))
        mat = self.rand_matrix(4, 4)
        self.assertEqual(Operator(mat.tolist()), Operator(mat))

    def test_data(self):
        """Test Operator representation string property."""
        mat = self.rand_matrix(2, 2)
        op = Operator(mat)
        assert_allclose(mat, op.data)

    def test_dim(self):
        """Test Operator dim property."""
        mat = self.rand_matrix(4, 4)
        self.assertEqual(Operator(mat).dim, (4, 4))
        self.assertEqual(Operator(mat, input_dims=[4], output_dims=[4]).dim, (4, 4))
        self.assertEqual(Operator(mat, input_dims=[2, 2], output_dims=[2, 2]).dim, (4, 4))

    def test_input_dims(self):
        """Test Operator input_dims method."""
        op = Operator(self.rand_matrix(2 * 3 * 4, 4 * 5), input_dims=[4, 5], output_dims=[2, 3, 4])
        self.assertEqual(op.input_dims(), (4, 5))
        self.assertEqual(op.input_dims(qargs=[0, 1]), (4, 5))
        self.assertEqual(op.input_dims(qargs=[1, 0]), (5, 4))
        self.assertEqual(op.input_dims(qargs=[0]), (4,))
        self.assertEqual(op.input_dims(qargs=[1]), (5,))

    def test_output_dims(self):
        """Test Operator output_dims method."""
        op = Operator(self.rand_matrix(2 * 3 * 4, 4 * 5), input_dims=[4, 5], output_dims=[2, 3, 4])
        self.assertEqual(op.output_dims(), (2, 3, 4))
        self.assertEqual(op.output_dims(qargs=[0, 1, 2]), (2, 3, 4))
        self.assertEqual(op.output_dims(qargs=[2, 1, 0]), (4, 3, 2))
        self.assertEqual(op.output_dims(qargs=[2, 0, 1]), (4, 2, 3))
        self.assertEqual(op.output_dims(qargs=[0]), (2,))
        self.assertEqual(op.output_dims(qargs=[1]), (3,))
        self.assertEqual(op.output_dims(qargs=[2]), (4,))
        self.assertEqual(op.output_dims(qargs=[0, 2]), (2, 4))
        self.assertEqual(op.output_dims(qargs=[2, 0]), (4, 2))

    def test_reshape(self):
        """Test Operator reshape method."""
        op = Operator(self.rand_matrix(8, 8))
        reshaped1 = op.reshape(input_dims=[8], output_dims=[8])
        reshaped2 = op.reshape(input_dims=[4, 2], output_dims=[2, 4])
        self.assertEqual(op.output_dims(), (2, 2, 2))
        self.assertEqual(op.input_dims(), (2, 2, 2))
        self.assertEqual(reshaped1.output_dims(), (8,))
        self.assertEqual(reshaped1.input_dims(), (8,))
        self.assertEqual(reshaped2.output_dims(), (2, 4))
        self.assertEqual(reshaped2.input_dims(), (4, 2))

    def test_reshape_num_qubits(self):
        """Test Operator reshape method with num_qubits."""
        op = Operator(self.rand_matrix(8, 8), input_dims=(4, 2), output_dims=(2, 4))
        reshaped = op.reshape(num_qubits=3)
        self.assertEqual(reshaped.num_qubits, 3)
        self.assertEqual(reshaped.output_dims(), (2, 2, 2))
        self.assertEqual(reshaped.input_dims(), (2, 2, 2))

    def test_reshape_raise(self):
        """Test Operator reshape method with invalid args."""
        op = Operator(self.rand_matrix(3, 3))
        self.assertRaises(QiskitError, op.reshape, num_qubits=2)

    def test_copy(self):
        """Test Operator copy method"""
        mat = np.eye(2)
        with self.subTest("Deep copy"):
            orig = Operator(mat)
            cpy = orig.copy()
            cpy._data[0, 0] = 0.0
            self.assertFalse(cpy == orig)
        with self.subTest("Shallow copy"):
            orig = Operator(mat)
            clone = copy.copy(orig)
            clone._data[0, 0] = 0.0
            self.assertTrue(clone == orig)

    def test_is_unitary(self):
        """Test is_unitary method."""
        # X-90 rotation
        X90 = la.expm(-1j * 0.5 * np.pi * np.array([[0, 1], [1, 0]]) / 2)
        self.assertTrue(Operator(X90).is_unitary())
        # Non-unitary should return false
        self.assertFalse(Operator([[1, 0], [0, 0]]).is_unitary())

    def test_to_operator(self):
        """Test to_operator method."""
        op1 = Operator(self.rand_matrix(4, 4))
        op2 = op1.to_operator()
        self.assertEqual(op1, op2)

    def test_conjugate(self):
        """Test conjugate method."""
        matr = self.rand_matrix(2, 4, real=True)
        mati = self.rand_matrix(2, 4, real=True)
        op = Operator(matr + 1j * mati)
        uni_conj = op.conjugate()
        self.assertEqual(uni_conj, Operator(matr - 1j * mati))

    def test_transpose(self):
        """Test transpose method."""
        matr = self.rand_matrix(2, 4, real=True)
        mati = self.rand_matrix(2, 4, real=True)
        op = Operator(matr + 1j * mati)
        uni_t = op.transpose()
        self.assertEqual(uni_t, Operator(matr.T + 1j * mati.T))

    def test_adjoint(self):
        """Test adjoint method."""
        matr = self.rand_matrix(2, 4, real=True)
        mati = self.rand_matrix(2, 4, real=True)
        op = Operator(matr + 1j * mati)
        uni_adj = op.adjoint()
        self.assertEqual(uni_adj, Operator(matr.T - 1j * mati.T))

    def test_compose_except(self):
        """Test compose different dimension exception"""
        self.assertRaises(QiskitError, Operator(np.eye(2)).compose, Operator(np.eye(3)))
        self.assertRaises(QiskitError, Operator(np.eye(2)).compose, 2)

    def test_compose(self):
        """Test compose method."""

        op1 = Operator(self.UX)
        op2 = Operator(self.UY)

        targ = Operator(np.dot(self.UY, self.UX))
        self.assertEqual(op1.compose(op2), targ)
        self.assertEqual(op1 & op2, targ)

        targ = Operator(np.dot(self.UX, self.UY))
        self.assertEqual(op2.compose(op1), targ)
        self.assertEqual(op2 & op1, targ)

    def test_dot(self):
        """Test dot method."""
        op1 = Operator(self.UY)
        op2 = Operator(self.UX)

        targ = Operator(np.dot(self.UY, self.UX))
        self.assertEqual(op1.dot(op2), targ)
        self.assertEqual(op1 * op2, targ)

        targ = Operator(np.dot(self.UX, self.UY))
        self.assertEqual(op2.dot(op1), targ)
        self.assertEqual(op2 * op1, targ)

    def test_compose_front(self):
        """Test front compose method."""

        opYX = Operator(self.UY).compose(Operator(self.UX), front=True)
        matYX = np.dot(self.UY, self.UX)
        self.assertEqual(opYX, Operator(matYX))

        opXY = Operator(self.UX).compose(Operator(self.UY), front=True)
        matXY = np.dot(self.UX, self.UY)
        self.assertEqual(opXY, Operator(matXY))

    def test_compose_subsystem(self):
        """Test subsystem compose method."""
        # 3-qubit operator
        mat = self.rand_matrix(8, 8)
        mat_a = self.rand_matrix(2, 2)
        mat_b = self.rand_matrix(2, 2)
        mat_c = self.rand_matrix(2, 2)
        op = Operator(mat)
        op1 = Operator(mat_a)
        op2 = Operator(np.kron(mat_b, mat_a))
        op3 = Operator(np.kron(mat_c, np.kron(mat_b, mat_a)))

        # op3 qargs=[0, 1, 2]
        targ = np.dot(np.kron(mat_c, np.kron(mat_b, mat_a)), mat)
        self.assertEqual(op.compose(op3, qargs=[0, 1, 2]), Operator(targ))
        self.assertEqual(op.compose(op3([0, 1, 2])), Operator(targ))
        self.assertEqual(op & op3([0, 1, 2]), Operator(targ))
        # op3 qargs=[2, 1, 0]
        targ = np.dot(np.kron(mat_a, np.kron(mat_b, mat_c)), mat)
        self.assertEqual(op.compose(op3, qargs=[2, 1, 0]), Operator(targ))
        self.assertEqual(op & op3([2, 1, 0]), Operator(targ))

        # op2 qargs=[0, 1]
        targ = np.dot(np.kron(np.eye(2), np.kron(mat_b, mat_a)), mat)
        self.assertEqual(op.compose(op2, qargs=[0, 1]), Operator(targ))
        self.assertEqual(op & op2([0, 1]), Operator(targ))
        # op2 qargs=[2, 0]
        targ = np.dot(np.kron(mat_a, np.kron(np.eye(2), mat_b)), mat)
        self.assertEqual(op.compose(op2, qargs=[2, 0]), Operator(targ))
        self.assertEqual(op & op2([2, 0]), Operator(targ))

        # op1 qargs=[0]
        targ = np.dot(np.kron(np.eye(4), mat_a), mat)
        self.assertEqual(op.compose(op1, qargs=[0]), Operator(targ))
        self.assertEqual(op & op1([0]), Operator(targ))
        # op1 qargs=[1]
        targ = np.dot(np.kron(np.eye(2), np.kron(mat_a, np.eye(2))), mat)
        self.assertEqual(op.compose(op1, qargs=[1]), Operator(targ))
        self.assertEqual(op & op1([1]), Operator(targ))
        # op1 qargs=[2]
        targ = np.dot(np.kron(mat_a, np.eye(4)), mat)
        self.assertEqual(op.compose(op1, qargs=[2]), Operator(targ))
        self.assertEqual(op & op1([2]), Operator(targ))

    def test_dot_subsystem(self):
        """Test subsystem dot method."""
        # 3-qubit operator
        mat = self.rand_matrix(8, 8)
        mat_a = self.rand_matrix(2, 2)
        mat_b = self.rand_matrix(2, 2)
        mat_c = self.rand_matrix(2, 2)
        op = Operator(mat)
        op1 = Operator(mat_a)
        op2 = Operator(np.kron(mat_b, mat_a))
        op3 = Operator(np.kron(mat_c, np.kron(mat_b, mat_a)))

        # op3 qargs=[0, 1, 2]
        targ = np.dot(mat, np.kron(mat_c, np.kron(mat_b, mat_a)))
        self.assertEqual(op.dot(op3, qargs=[0, 1, 2]), Operator(targ))
        self.assertEqual(op.dot(op3([0, 1, 2])), Operator(targ))
        # op3 qargs=[2, 1, 0]
        targ = np.dot(mat, np.kron(mat_a, np.kron(mat_b, mat_c)))
        self.assertEqual(op.dot(op3, qargs=[2, 1, 0]), Operator(targ))
        self.assertEqual(op.dot(op3([2, 1, 0])), Operator(targ))

        # op2 qargs=[0, 1]
        targ = np.dot(mat, np.kron(np.eye(2), np.kron(mat_b, mat_a)))
        self.assertEqual(op.dot(op2, qargs=[0, 1]), Operator(targ))
        self.assertEqual(op.dot(op2([0, 1])), Operator(targ))
        # op2 qargs=[2, 0]
        targ = np.dot(mat, np.kron(mat_a, np.kron(np.eye(2), mat_b)))
        self.assertEqual(op.dot(op2, qargs=[2, 0]), Operator(targ))
        self.assertEqual(op.dot(op2([2, 0])), Operator(targ))

        # op1 qargs=[0]
        targ = np.dot(mat, np.kron(np.eye(4), mat_a))
        self.assertEqual(op.dot(op1, qargs=[0]), Operator(targ))
        self.assertEqual(op.dot(op1([0])), Operator(targ))
        # op1 qargs=[1]
        targ = np.dot(mat, np.kron(np.eye(2), np.kron(mat_a, np.eye(2))))
        self.assertEqual(op.dot(op1, qargs=[1]), Operator(targ))
        self.assertEqual(op.dot(op1([1])), Operator(targ))
        # op1 qargs=[2]
        targ = np.dot(mat, np.kron(mat_a, np.eye(4)))
        self.assertEqual(op.dot(op1, qargs=[2]), Operator(targ))
        self.assertEqual(op.dot(op1([2])), Operator(targ))

    def test_compose_front_subsystem(self):
        """Test subsystem front compose method."""
        # 3-qubit operator
        mat = self.rand_matrix(8, 8)
        mat_a = self.rand_matrix(2, 2)
        mat_b = self.rand_matrix(2, 2)
        mat_c = self.rand_matrix(2, 2)
        op = Operator(mat)
        op1 = Operator(mat_a)
        op2 = Operator(np.kron(mat_b, mat_a))
        op3 = Operator(np.kron(mat_c, np.kron(mat_b, mat_a)))

        # op3 qargs=[0, 1, 2]
        targ = np.dot(mat, np.kron(mat_c, np.kron(mat_b, mat_a)))
        self.assertEqual(op.compose(op3, qargs=[0, 1, 2], front=True), Operator(targ))
        # op3 qargs=[2, 1, 0]
        targ = np.dot(mat, np.kron(mat_a, np.kron(mat_b, mat_c)))
        self.assertEqual(op.compose(op3, qargs=[2, 1, 0], front=True), Operator(targ))

        # op2 qargs=[0, 1]
        targ = np.dot(mat, np.kron(np.eye(2), np.kron(mat_b, mat_a)))
        self.assertEqual(op.compose(op2, qargs=[0, 1], front=True), Operator(targ))
        # op2 qargs=[2, 0]
        targ = np.dot(mat, np.kron(mat_a, np.kron(np.eye(2), mat_b)))
        self.assertEqual(op.compose(op2, qargs=[2, 0], front=True), Operator(targ))

        # op1 qargs=[0]
        targ = np.dot(mat, np.kron(np.eye(4), mat_a))
        self.assertEqual(op.compose(op1, qargs=[0], front=True), Operator(targ))

        # op1 qargs=[1]
        targ = np.dot(mat, np.kron(np.eye(2), np.kron(mat_a, np.eye(2))))
        self.assertEqual(op.compose(op1, qargs=[1], front=True), Operator(targ))

        # op1 qargs=[2]
        targ = np.dot(mat, np.kron(mat_a, np.eye(4)))
        self.assertEqual(op.compose(op1, qargs=[2], front=True), Operator(targ))

    def test_power(self):
        """Test power method."""
        X90 = la.expm(-1j * 0.5 * np.pi * np.array([[0, 1], [1, 0]]) / 2)
        op = Operator(X90)
        self.assertEqual(op.power(2), Operator([[0, -1j], [-1j, 0]]))
        self.assertEqual(op.power(4), Operator(-1 * np.eye(2)))
        self.assertEqual(op.power(8), Operator(np.eye(2)))

    def test_expand(self):
        """Test expand method."""
        mat1 = self.UX
        mat2 = np.eye(3, dtype=complex)

        mat21 = np.kron(mat2, mat1)
        op21 = Operator(mat1).expand(Operator(mat2))
        self.assertEqual(op21.dim, (6, 6))
        assert_allclose(op21.data, Operator(mat21).data)

        mat12 = np.kron(mat1, mat2)
        op12 = Operator(mat2).expand(Operator(mat1))
        self.assertEqual(op12.dim, (6, 6))
        assert_allclose(op12.data, Operator(mat12).data)

    def test_tensor(self):
        """Test tensor method."""
        mat1 = self.UX
        mat2 = np.eye(3, dtype=complex)

        mat21 = np.kron(mat2, mat1)
        op21 = Operator(mat2).tensor(Operator(mat1))
        self.assertEqual(op21.dim, (6, 6))
        assert_allclose(op21.data, Operator(mat21).data)

        mat12 = np.kron(mat1, mat2)
        op12 = Operator(mat1).tensor(Operator(mat2))
        self.assertEqual(op12.dim, (6, 6))
        assert_allclose(op12.data, Operator(mat12).data)

    def test_power_except(self):
        """Test power method raises exceptions if not square."""
        op = Operator(self.rand_matrix(2, 3))
        # Non-integer power raises error
        self.assertRaises(QiskitError, op.power, 0.5)

    def test_add(self):
        """Test add method."""
        mat1 = self.rand_matrix(4, 4)
        mat2 = self.rand_matrix(4, 4)
        op1 = Operator(mat1)
        op2 = Operator(mat2)
        self.assertEqual(op1._add(op2), Operator(mat1 + mat2))
        self.assertEqual(op1 + op2, Operator(mat1 + mat2))
        self.assertEqual(op1 - op2, Operator(mat1 - mat2))

    def test_add_except(self):
        """Test add method raises exceptions."""
        op1 = Operator(self.rand_matrix(2, 2))
        op2 = Operator(self.rand_matrix(3, 3))
        self.assertRaises(QiskitError, op1._add, op2)

    def test_add_qargs(self):
        """Test add method with qargs."""
        mat = self.rand_matrix(8, 8)
        mat0 = self.rand_matrix(2, 2)
        mat1 = self.rand_matrix(2, 2)

        op = Operator(mat)
        op0 = Operator(mat0)
        op01 = Operator(np.kron(mat1, mat0))

        with self.subTest(msg="qargs=[0]"):
            value = op + op0([0])
            target = op + Operator(np.kron(np.eye(4), mat0))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[1]"):
            value = op + op0([1])
            target = op + Operator(np.kron(np.kron(np.eye(2), mat0), np.eye(2)))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[2]"):
            value = op + op0([2])
            target = op + Operator(np.kron(mat0, np.eye(4)))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[0, 1]"):
            value = op + op01([0, 1])
            target = op + Operator(np.kron(np.eye(2), np.kron(mat1, mat0)))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[1, 0]"):
            value = op + op01([1, 0])
            target = op + Operator(np.kron(np.eye(2), np.kron(mat0, mat1)))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[0, 2]"):
            value = op + op01([0, 2])
            target = op + Operator(np.kron(mat1, np.kron(np.eye(2), mat0)))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[2, 0]"):
            value = op + op01([2, 0])
            target = op + Operator(np.kron(mat0, np.kron(np.eye(2), mat1)))
            self.assertEqual(value, target)

    def test_sub_qargs(self):
        """Test subtract method with qargs."""
        mat = self.rand_matrix(8, 8)
        mat0 = self.rand_matrix(2, 2)
        mat1 = self.rand_matrix(2, 2)

        op = Operator(mat)
        op0 = Operator(mat0)
        op01 = Operator(np.kron(mat1, mat0))

        with self.subTest(msg="qargs=[0]"):
            value = op - op0([0])
            target = op - Operator(np.kron(np.eye(4), mat0))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[1]"):
            value = op - op0([1])
            target = op - Operator(np.kron(np.kron(np.eye(2), mat0), np.eye(2)))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[2]"):
            value = op - op0([2])
            target = op - Operator(np.kron(mat0, np.eye(4)))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[0, 1]"):
            value = op - op01([0, 1])
            target = op - Operator(np.kron(np.eye(2), np.kron(mat1, mat0)))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[1, 0]"):
            value = op - op01([1, 0])
            target = op - Operator(np.kron(np.eye(2), np.kron(mat0, mat1)))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[0, 2]"):
            value = op - op01([0, 2])
            target = op - Operator(np.kron(mat1, np.kron(np.eye(2), mat0)))
            self.assertEqual(value, target)

        with self.subTest(msg="qargs=[2, 0]"):
            value = op - op01([2, 0])
            target = op - Operator(np.kron(mat0, np.kron(np.eye(2), mat1)))
            self.assertEqual(value, target)

    def test_multiply(self):
        """Test multiply method."""
        mat = self.rand_matrix(4, 4)
        val = np.exp(5j)
        op = Operator(mat)
        self.assertEqual(op._multiply(val), Operator(val * mat))
        self.assertEqual(val * op, Operator(val * mat))

    def test_multiply_except(self):
        """Test multiply method raises exceptions."""
        op = Operator(self.rand_matrix(2, 2))
        self.assertRaises(QiskitError, op._multiply, "s")
        self.assertRaises(QiskitError, op.__rmul__, "s")
        self.assertRaises(QiskitError, op._multiply, op)
        self.assertRaises(QiskitError, op.__rmul__, op)

    def test_negate(self):
        """Test negate method"""
        mat = self.rand_matrix(4, 4)
        op = Operator(mat)
        self.assertEqual(-op, Operator(-1 * mat))

    def test_equiv(self):
        """Test negate method"""
        mat = np.diag([1, np.exp(1j * np.pi / 2)])
        phase = np.exp(-1j * np.pi / 4)
        op = Operator(mat)
        self.assertTrue(op.equiv(phase * mat))
        self.assertTrue(op.equiv(Operator(phase * mat)))
        self.assertFalse(op.equiv(2 * mat))

    def test_reverse_qargs(self):
        """Test reverse_qargs method"""
        circ1 = QFT(5)
        circ2 = circ1.reverse_bits()

        state1 = Operator(circ1)
        state2 = Operator(circ2)
        self.assertEqual(state1.reverse_qargs(), state2)


if __name__ == "__main__":
    unittest.main()
