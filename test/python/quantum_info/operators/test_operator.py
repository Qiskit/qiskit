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

"""Tests for Operator matrix linear operator class."""

import unittest
import numpy as np
import scipy.linalg as la

from qiskit import QiskitError
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions.standard import HGate, CHGate, CnotGate
from qiskit.test import QiskitTestCase
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.predicates import matrix_equal


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
        psi = np.random.rand(n) + 1j * np.random.rand(n)
        rho = np.outer(psi, psi.conj())
        rho /= np.trace(rho)
        return rho

    @classmethod
    def rand_matrix(cls, rows, cols=None, real=False):
        """Return a random matrix."""
        if cols is None:
            cols = rows
        if real:
            return np.random.rand(rows, cols)
        return np.random.rand(rows, cols) + 1j * np.random.rand(rows, cols)

    def simple_circuit_no_measure(self):
        """Return a unitary circuit and the corresponding unitary array."""
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.h(qr[0])
        circ.x(qr[1])
        target = Operator(np.kron(self.UX, self.UH))
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

    def assertAllClose(self,
                       obj1,
                       obj2,
                       rtol=1e-5,
                       atol=1e-6,
                       equal_nan=False,
                       msg=None):
        """Assert two objects are equal using Numpy.allclose."""
        comparison = np.allclose(
            obj1, obj2, rtol=rtol, atol=atol, equal_nan=equal_nan)
        if msg is None:
            msg = ''
        msg += '({} != {})'.format(obj1, obj2)
        self.assertTrue(comparison, msg=msg)


class TestOperator(OperatorTestCase):
    """Tests for Operator linear operator class."""

    def test_init_array_qubit(self):
        """Test subsystem initialization from N-qubit array."""
        # Test automatic inference of qubit subsystems
        mat = self.rand_matrix(8, 8)
        op = Operator(mat)
        self.assertAllClose(op.data, mat)
        self.assertEqual(op.dim, (8, 8))
        self.assertEqual(op.input_dims(), (2, 2, 2))
        self.assertEqual(op.output_dims(), (2, 2, 2))

        op = Operator(mat, input_dims=8, output_dims=8)
        self.assertAllClose(op.data, mat)
        self.assertEqual(op.dim, (8, 8))
        self.assertEqual(op.input_dims(), (2, 2, 2))
        self.assertEqual(op.output_dims(), (2, 2, 2))

    def test_init_array(self):
        """Test initialization from array."""
        mat = np.eye(3)
        op = Operator(mat)
        self.assertAllClose(op.data, mat)
        self.assertEqual(op.dim, (3, 3))
        self.assertEqual(op.input_dims(), (3,))
        self.assertEqual(op.output_dims(), (3,))

        mat = self.rand_matrix(2 * 3 * 4, 4 * 5)
        op = Operator(mat, input_dims=[4, 5], output_dims=[2, 3, 4])
        self.assertAllClose(op.data, mat)
        self.assertEqual(op.dim, (4 * 5, 2 * 3 * 4))
        self.assertEqual(op.input_dims(), (4, 5))
        self.assertEqual(op.output_dims(), (2, 3, 4))

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
        circuit, target = self.simple_circuit_no_measure()
        op = Operator(circuit)
        target = Operator(target)
        self.assertEqual(op, target)

    def test_instruction_init(self):
        """Test initialization from a circuit."""
        gate = CnotGate()
        op = Operator(gate).data
        target = gate.to_matrix()
        global_phase_equivalent = matrix_equal(op, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

        gate = CHGate()
        op = Operator(gate).data
        had = HGate().to_matrix()
        target = np.kron(had, np.diag([0, 1])) + np.kron(
            np.eye(2), np.diag([1, 0]))
        global_phase_equivalent = matrix_equal(op, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

    def test_circuit_init_except(self):
        """Test initialization from circuit with measure raises exception."""
        circuit = self.simple_circuit_with_measure()
        self.assertRaises(QiskitError, Operator, circuit)

    def test_equal(self):
        """Test __eq__ method"""
        mat = self.rand_matrix(2, 2, real=True)
        self.assertEqual(Operator(np.array(mat, dtype=complex)),
                         Operator(mat))
        mat = self.rand_matrix(4, 4)
        self.assertEqual(Operator(mat.tolist()),
                         Operator(mat))

    def test_rep(self):
        """Test Operator representation string property."""
        op = Operator(self.rand_matrix(2, 2))
        self.assertEqual(op.rep, 'Operator')

    def test_data(self):
        """Test Operator representation string property."""
        mat = self.rand_matrix(2, 2)
        op = Operator(mat)
        self.assertAllClose(mat, op.data)

    def test_dim(self):
        """Test Operator dim property."""
        mat = self.rand_matrix(4, 4)
        self.assertEqual(Operator(mat).dim, (4, 4))
        self.assertEqual(Operator(mat, input_dims=[4], output_dims=[4]).dim, (4, 4))
        self.assertEqual(Operator(mat, input_dims=[2, 2], output_dims=[2, 2]).dim, (4, 4))

    def test_input_dims(self):
        """Test Operator input_dims method."""
        op = Operator(self.rand_matrix(2 * 3 * 4, 4 * 5),
                      input_dims=[4, 5], output_dims=[2, 3, 4])
        self.assertEqual(op.input_dims(), (4, 5))
        self.assertEqual(op.input_dims(qargs=[0, 1]), (4, 5))
        self.assertEqual(op.input_dims(qargs=[1, 0]), (5, 4))
        self.assertEqual(op.input_dims(qargs=[0]), (4,))
        self.assertEqual(op.input_dims(qargs=[1]), (5,))

    def test_output_dims(self):
        """Test Operator output_dims method."""
        op = Operator(self.rand_matrix(2 * 3 * 4, 4 * 5),
                      input_dims=[4, 5], output_dims=[2, 3, 4])
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
        """Test Operator _reshape method."""
        op = Operator(self.rand_matrix(8, 8))
        self.assertEqual(op.output_dims(), (2, 2, 2))
        self.assertEqual(op.input_dims(), (2, 2, 2))
        op._reshape(input_dims=[8], output_dims=[8])
        self.assertEqual(op.output_dims(), (8,))
        self.assertEqual(op.input_dims(), (8,))
        op._reshape(input_dims=[4, 2], output_dims=[2, 4])
        self.assertEqual(op.output_dims(), (2, 4))
        self.assertEqual(op.input_dims(), (4, 2))

    def test_copy(self):
        """Test Operator copy method"""
        mat = np.eye(2)
        orig = Operator(mat)
        cpy = orig.copy()
        cpy._data[0, 0] = 0.0
        self.assertFalse(cpy == orig)

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

    def test_evolve(self):
        """Test _evolve method."""
        # Test hadamard
        op = Operator(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
        target_psi = np.array([1, 1]) / np.sqrt(2)
        target_rho = np.array([[0.5, 0.5], [0.5, 0.5]])
        # Test list vector evolve
        self.assertAllClose(op._evolve([1, 0]), target_psi)
        # Test np.array vector evolve
        self.assertAllClose(op._evolve(np.array([1, 0])), target_psi)
        # Test list density matrix evolve
        self.assertAllClose(op._evolve([[1, 0], [0, 0]]), target_rho)
        # Test np.array density matrix evolve
        self.assertAllClose(
            op._evolve(np.array([[1, 0], [0, 0]])), target_rho)

    def test_evolve_subsystem(self):
        """Test subsystem _evolve method."""
        # Test evolving single-qubit of 3-qubit system
        mat = self.rand_matrix(2, 2)
        op = Operator(mat)
        psi = self.rand_matrix(1, 8).flatten()
        rho = self.rand_rho(8)

        # Evolve on qubit 0
        mat0 = np.kron(np.eye(4), mat)
        psi0_targ = np.dot(mat0, psi)
        self.assertAllClose(op._evolve(psi, qargs=[0]), psi0_targ)
        rho0_targ = np.dot(np.dot(mat0, rho), np.conj(mat0.T))
        self.assertAllClose(op._evolve(rho, qargs=[0]), rho0_targ)

        # Evolve on qubit 1
        mat1 = np.kron(np.kron(np.eye(2), mat), np.eye(2))
        psi1_targ = np.dot(mat1, psi)
        self.assertAllClose(op._evolve(psi, qargs=[1]), psi1_targ)
        rho1_targ = np.dot(np.dot(mat1, rho), np.conj(mat1.T))
        self.assertAllClose(op._evolve(rho, qargs=[1]), rho1_targ)

        # Evolve on qubit 2
        mat2 = np.kron(mat, np.eye(4))
        psi2_targ = np.dot(mat2, psi)
        self.assertAllClose(op._evolve(psi, qargs=[2]), psi2_targ)
        rho2_targ = np.dot(np.dot(mat2, rho), np.conj(mat2.T))
        self.assertAllClose(op._evolve(rho, qargs=[2]), rho2_targ)

        # Test 2-qubit evolution
        mat_a = self.rand_matrix(2, 2)
        mat_b = self.rand_matrix(2, 2)
        op = Operator(np.kron(mat_b, mat_a))
        psi = self.rand_matrix(1, 8).flatten()
        rho = self.rand_rho(8)

        # Evolve on qubits [0, 2]
        mat02 = np.kron(mat_b, np.kron(np.eye(2), mat_a))
        psi02_targ = np.dot(mat02, psi)
        self.assertAllClose(op._evolve(psi, qargs=[0, 2]), psi02_targ)
        rho02_targ = np.dot(np.dot(mat02, rho), np.conj(mat02.T))
        self.assertAllClose(op._evolve(rho, qargs=[0, 2]), rho02_targ)

        # Evolve on qubits [2, 0]
        mat20 = np.kron(mat_a, np.kron(np.eye(2), mat_b))
        psi20_targ = np.dot(mat20, psi)
        self.assertAllClose(op._evolve(psi, qargs=[2, 0]), psi20_targ)
        rho20_targ = np.dot(np.dot(mat20, rho), np.conj(mat20.T))
        self.assertAllClose(op._evolve(rho, qargs=[2, 0]), rho20_targ)

        # Test evolve on 3-qubits
        mat_a = self.rand_matrix(2, 2)
        mat_b = self.rand_matrix(2, 2)
        mat_c = self.rand_matrix(2, 2)
        op = Operator(np.kron(mat_c, np.kron(mat_b, mat_a)))
        psi = self.rand_matrix(1, 8).flatten()
        rho = self.rand_rho(8)

        # Evolve on qubits [0, 1, 2]
        mat012 = np.kron(mat_c, np.kron(mat_b, mat_a))
        psi012_targ = np.dot(mat012, psi)
        self.assertAllClose(op._evolve(psi, qargs=[0, 1, 2]), psi012_targ)
        rho012_targ = np.dot(np.dot(mat012, rho), np.conj(mat012.T))
        self.assertAllClose(op._evolve(rho, qargs=[0, 1, 2]), rho012_targ)

        # Evolve on qubits [2, 1, 0]
        mat210 = np.kron(mat_a, np.kron(mat_b, mat_c))
        psi210_targ = np.dot(mat210, psi)
        self.assertAllClose(op._evolve(psi, qargs=[2, 1, 0]), psi210_targ)
        rho210_targ = np.dot(np.dot(mat210, rho), np.conj(mat210.T))
        self.assertAllClose(op._evolve(rho, qargs=[2, 1, 0]), rho210_targ)

    def test_evolve_rand(self):
        """Test evolve method on random state."""
        mat = self.rand_matrix(4, 4)
        rho = self.rand_rho(4)
        target_rho = np.dot(np.dot(mat, rho), np.conj(mat.T))
        op = Operator(mat)
        self.assertAllClose(op._evolve(rho), target_rho)

    def test_conjugate(self):
        """Test conjugate method."""
        matr = self.rand_matrix(2, 2, real=True)
        mati = self.rand_matrix(2, 2, real=True)
        op = Operator(matr + 1j * mati)
        uni_conj = op.conjugate()
        self.assertEqual(uni_conj, Operator(matr - 1j * mati))

    def test_transpose(self):
        """Test transpose method."""
        matr = self.rand_matrix(2, 2, real=True)
        mati = self.rand_matrix(2, 2, real=True)
        op = Operator(matr + 1j * mati)
        uni_t = op.transpose()
        self.assertEqual(uni_t, Operator(matr.T + 1j * mati.T))

    def test_adjoint(self):
        """Test adjoint method."""
        matr = self.rand_matrix(2, 2, real=True)
        mati = self.rand_matrix(2, 2, real=True)
        op = Operator(matr + 1j * mati)
        uni_adj = op.adjoint()
        self.assertEqual(uni_adj, Operator(matr.T - 1j * mati.T))

    def test_compose_except(self):
        """Test compose different dimension exception"""
        self.assertRaises(QiskitError,
                          Operator(np.eye(2)).compose,
                          Operator(np.eye(3)))
        self.assertRaises(QiskitError, Operator(np.eye(2)).compose, 2)

    def test_compose(self):
        """Test compose method."""

        op1 = Operator(self.UX)
        op2 = Operator(self.UY)

        targ = Operator(np.dot(self.UY, self.UX))
        self.assertEqual(op1.compose(op2), targ)
        self.assertEqual(op1 @ op2, targ)

        targ = Operator(np.dot(self.UX, self.UY))
        self.assertEqual(op2.compose(op1), targ)
        self.assertEqual(op2 @ op1, targ)

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
        # op3 qargs=[2, 1, 0]
        targ = np.dot(np.kron(mat_a, np.kron(mat_b, mat_c)), mat)
        self.assertEqual(op.compose(op3, qargs=[2, 1, 0]), Operator(targ))

        # op2 qargs=[0, 1]
        targ = np.dot(np.kron(np.eye(2), np.kron(mat_b, mat_a)), mat)
        self.assertEqual(op.compose(op2, qargs=[0, 1]), Operator(targ))
        # op2 qargs=[2, 0]
        targ = np.dot(np.kron(mat_a, np.kron(np.eye(2), mat_b)), mat)
        self.assertEqual(op.compose(op2, qargs=[2, 0]), Operator(targ))

        # op1 qargs=[0]
        targ = np.dot(np.kron(np.eye(4), mat_a), mat)
        self.assertEqual(op.compose(op1, qargs=[0]), Operator(targ))

        # op1 qargs=[1]
        targ = np.dot(np.kron(np.eye(2), np.kron(mat_a, np.eye(2))), mat)
        self.assertEqual(op.compose(op1, qargs=[1]), Operator(targ))

        # op1 qargs=[2]
        targ = np.dot(np.kron(mat_a, np.eye(4)), mat)
        self.assertEqual(op.compose(op1, qargs=[2]), Operator(targ))

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
        self.assertAllClose(op21.data, Operator(mat21).data)

        mat12 = np.kron(mat1, mat2)
        op12 = Operator(mat2).expand(Operator(mat1))
        self.assertEqual(op12.dim, (6, 6))
        self.assertAllClose(op12.data, Operator(mat12).data)

    def test_tensor(self):
        """Test tensor method."""
        mat1 = self.UX
        mat2 = np.eye(3, dtype=complex)

        mat21 = np.kron(mat2, mat1)
        op21 = Operator(mat2).tensor(Operator(mat1))
        self.assertEqual(op21.dim, (6, 6))
        self.assertAllClose(op21.data, Operator(mat21).data)

        mat12 = np.kron(mat1, mat2)
        op12 = Operator(mat1).tensor(Operator(mat2))
        self.assertEqual(op12.dim, (6, 6))
        self.assertAllClose(op12.data, Operator(mat12).data)

    def test_power_except(self):
        """Test power method raises exceptions."""
        op = Operator(self.rand_matrix(3, 3))
        # Non-integer power raises error
        self.assertRaises(QiskitError, op.power, 0.5)

    def test_add(self):
        """Test add method."""
        mat1 = self.rand_matrix(4, 4)
        mat2 = self.rand_matrix(4, 4)
        op1 = Operator(mat1)
        op2 = Operator(mat2)
        self.assertEqual(op1.add(op2), Operator(mat1 + mat2))
        self.assertEqual(op1 + op2, Operator(mat1 + mat2))

    def test_add_except(self):
        """Test add method raises exceptions."""
        op1 = Operator(self.rand_matrix(2, 2))
        op2 = Operator(self.rand_matrix(3, 3))
        self.assertRaises(QiskitError, op1.add, op2)

    def test_subtract(self):
        """Test subtract method."""
        mat1 = self.rand_matrix(4, 4)
        mat2 = self.rand_matrix(4, 4)
        op1 = Operator(mat1)
        op2 = Operator(mat2)
        self.assertEqual(op1.subtract(op2), Operator(mat1 - mat2))
        self.assertEqual(op1 - op2, Operator(mat1 - mat2))

    def test_subtract_except(self):
        """Test subtract method raises exceptions."""
        op1 = Operator(self.rand_matrix(2, 2))
        op2 = Operator(self.rand_matrix(3, 3))
        self.assertRaises(QiskitError, op1.subtract, op2)

    def test_multiply(self):
        """Test multiply method."""
        mat = self.rand_matrix(4, 4)
        val = np.exp(5j)
        op = Operator(mat)
        self.assertEqual(op.multiply(val), Operator(val * mat))
        self.assertEqual(val * op, Operator(val * mat))

    def test_multiply_except(self):
        """Test multiply method raises exceptions."""
        op = Operator(self.rand_matrix(2, 2))
        self.assertRaises(QiskitError, op.multiply, 's')
        self.assertRaises(QiskitError, op.multiply, op)

    def test_negate(self):
        """Test negate method"""
        mat = self.rand_matrix(4, 4)
        op = Operator(mat)
        self.assertEqual(-op, Operator(-1 * mat))


if __name__ == '__main__':
    unittest.main()
