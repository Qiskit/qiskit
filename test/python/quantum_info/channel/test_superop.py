# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring

"""Tests for SuperOp quantum channel representation class."""

import unittest
import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info.operators.channel import SuperOp
from .base import ChannelTestCase


class TestSuperOp(ChannelTestCase):
    """Tests for SuperOp channel representation."""

    def test_init(self):
        """Test initialization"""
        chan = SuperOp(self.sopI)
        self.assertAllClose(chan.data, self.sopI)
        self.assertEqual(chan.dims, (2, 2))

        mat = np.zeros((4, 16))
        chan = SuperOp(mat)
        self.assertAllClose(chan.data, mat)
        self.assertEqual(chan.dims, (4, 2))

        chan = SuperOp(mat.T)
        self.assertAllClose(chan.data, mat.T)
        self.assertEqual(chan.dims, (2, 4))

        # Wrong input or output dims should raise exception
        self.assertRaises(QiskitError, SuperOp, mat, input_dim=4, output_dim=4)

    def test_equal(self):
        """Test __eq__ method"""
        mat = self.rand_matrix(4, 4)
        self.assertEqual(SuperOp(mat), SuperOp(mat))

    def test_copy(self):
        """Test copy method"""
        mat = np.eye(4)
        orig = SuperOp(mat)
        cpy = orig.copy()
        cpy._data[0, 0] = 0.0
        self.assertFalse(cpy == orig)

    def test_evolve(self):
        """Test evolve method."""
        input_psi = [0, 1]
        input_rho = [[0, 0], [0, 1]]
        # Identity channel
        chan = SuperOp(self.sopI)
        target_rho = np.array([[0, 0], [0, 1]])
        self.assertAllClose(chan._evolve(input_psi), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_psi)), target_rho)
        self.assertAllClose(chan._evolve(input_rho), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_rho)), target_rho)

        # Hadamard channel
        mat = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        chan = SuperOp(np.kron(mat.conj(), mat))
        target_rho = np.array([[1, -1], [-1, 1]]) / 2
        self.assertAllClose(chan._evolve(input_psi), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_psi)), target_rho)
        self.assertAllClose(chan._evolve(input_rho), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_rho)), target_rho)

        # Completely depolarizing channel
        chan = SuperOp(self.depol_sop(1))
        target_rho = np.eye(2) / 2
        self.assertAllClose(chan._evolve(input_psi), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_psi)), target_rho)
        self.assertAllClose(chan._evolve(input_rho), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_rho)), target_rho)

    def test_is_cptp(self):
        """Test is_cptp method."""
        self.assertTrue(SuperOp(self.depol_sop(0.25)).is_cptp())
        # Non-CPTP should return false
        self.assertFalse(SuperOp(1.25 * self.sopI - 0.25 * self.depol_sop(1)).is_cptp())

    def test_conjugate(self):
        """Test conjugate method."""
        mat = self.rand_matrix(4, 4)
        chan = SuperOp(mat)
        targ = SuperOp(np.conjugate(mat))
        self.assertEqual(chan.conjugate(), targ)

    def test_conjugate_inplace(self):
        """Test inplace conjugate method."""
        mat = self.rand_matrix(4, 4)
        chan = SuperOp(mat)
        chan.conjugate(inplace=True)
        targ = SuperOp(np.conjugate(mat))
        self.assertEqual(chan, targ)

    def test_transpose(self):
        """Test transpose method."""
        mat = self.rand_matrix(4, 4)
        chan = SuperOp(mat)
        targ = SuperOp(np.transpose(mat))
        self.assertEqual(chan.transpose(), targ)

    def test_transpose_inplace(self):
        """Test inplace transpose method."""
        mat = self.rand_matrix(4, 4)
        chan = SuperOp(mat)
        chan.transpose(inplace=True)
        targ = SuperOp(np.transpose(mat))
        self.assertEqual(chan, targ)

    def test_adjoint(self):
        """Test adjoint method."""
        mat = self.rand_matrix(4, 4)
        chan = SuperOp(mat)
        targ = SuperOp(np.transpose(np.conj(mat)))
        self.assertEqual(chan.adjoint(), targ)

    def test_adjoint_inplace(self):
        """Test inplace adjoint method."""
        mat = self.rand_matrix(4, 4)
        chan = SuperOp(mat)
        chan.adjoint(inplace=True)
        targ = SuperOp(np.transpose(np.conj(mat)))
        self.assertEqual(chan, targ)

    def test_compose_except(self):
        """Test compose different dimension exception"""
        self.assertRaises(QiskitError, SuperOp(np.eye(4)).compose, SuperOp(np.eye(16)))
        self.assertRaises(QiskitError, SuperOp(np.eye(4)).compose, np.eye(4))
        self.assertRaises(QiskitError, SuperOp(np.eye(4)).compose, 2)

    def test_compose(self):
        """Test compose method."""
        # UnitaryChannel evolution
        chan1 = SuperOp(self.sopX)
        chan2 = SuperOp(self.sopY)
        chan = chan1.compose(chan2)
        targ = SuperOp(self.sopZ)
        self.assertEqual(chan, targ)

        # 50% depolarizing channel
        chan1 = SuperOp(self.depol_sop(0.5))
        chan = chan1.compose(chan1)
        targ = SuperOp(self.depol_sop(0.75))
        self.assertEqual(chan, targ)

        # Random superoperator
        mat1 = self.rand_matrix(4, 4)
        mat2 = self.rand_matrix(4, 4)
        chan1 = SuperOp(mat1)
        chan2 = SuperOp(mat2)
        targ = SuperOp(np.dot(mat2, mat1))
        self.assertEqual(chan1.compose(chan2), targ)
        self.assertEqual(chan1 @ chan2, targ)
        targ = SuperOp(np.dot(mat1, mat2))
        self.assertEqual(chan2.compose(chan1), targ)
        self.assertEqual(chan2 @ chan1, targ)

        # Compose different dimensions
        chan1 = SuperOp(self.rand_matrix(16, 4), input_dim=2, output_dim=4)
        chan2 = SuperOp(self.rand_matrix(4, 16), output_dim=2)
        chan = chan1.compose(chan2)
        self.assertEqual(chan.dims, (2, 2))
        chan = chan2.compose(chan1)
        self.assertEqual(chan.dims, (4, 4))

    def test_compose_inplace(self):
        """Test inplace compose method."""
        # UnitaryChannel evolution
        chan1 = SuperOp(self.sopX)
        chan2 = SuperOp(self.sopY)
        targ = SuperOp(self.sopZ)
        chan1.compose(chan2, inplace=True)
        self.assertEqual(chan1, targ)

        # 50% depolarizing channel
        chan1 = SuperOp(self.depol_sop(0.5))
        chan1.compose(chan1, inplace=True)
        targ = SuperOp(self.depol_sop(0.75))
        self.assertEqual(chan1, targ)

        # Random superoperator
        mat1 = self.rand_matrix(4, 4)
        mat2 = self.rand_matrix(4, 4)

        targ = SuperOp(np.dot(mat2, mat1))
        chan1 = SuperOp(mat1)
        chan2 = SuperOp(mat2)
        chan1.compose(chan2, inplace=True)
        self.assertEqual(chan1, targ)
        chan1 = SuperOp(mat1)
        chan2 = SuperOp(mat2)
        chan1 @= chan2
        self.assertEqual(chan1, targ)

        targ = SuperOp(np.dot(mat1, mat2))
        chan1 = SuperOp(mat1)
        chan2 = SuperOp(mat2)
        chan2.compose(chan1, inplace=True)
        self.assertEqual(chan2, targ)
        chan1 = SuperOp(mat1)
        chan2 = SuperOp(mat2)
        chan2 @= chan1
        self.assertEqual(chan2, targ)

        # Compose different dimensions
        chan1 = SuperOp(self.rand_matrix(16, 4), input_dim=2, output_dim=4)
        chan2 = SuperOp(self.rand_matrix(4, 16), output_dim=2)
        chan1.compose(chan2, inplace=True)
        self.assertEqual(chan1.dims, (2, 2))
        chan1 = SuperOp(self.rand_matrix(16, 4), input_dim=2, output_dim=4)
        chan2 = SuperOp(self.rand_matrix(4, 16), output_dim=2)
        chan2.compose(chan1, inplace=True)
        self.assertEqual(chan2.dims, (4, 4))

    def test_compose_front(self):
        """Test front compose method."""
        # UnitaryChannel evolution
        chan1 = SuperOp(self.sopX)
        chan2 = SuperOp(self.sopY)
        chan = chan1.compose(chan2, front=True)
        targ = SuperOp(self.sopZ)
        self.assertEqual(chan, targ)

        # 50% depolarizing channel
        chan1 = SuperOp(self.depol_sop(0.5))
        chan = chan1.compose(chan1, front=True)
        targ = SuperOp(self.depol_sop(0.75))
        self.assertEqual(chan, targ)

        # Random superoperator
        mat1 = self.rand_matrix(4, 4)
        mat2 = self.rand_matrix(4, 4)
        chan1 = SuperOp(mat1)
        chan2 = SuperOp(mat2)
        targ = SuperOp(np.dot(mat2, mat1))
        self.assertEqual(chan2.compose(chan1, front=True), targ)
        targ = SuperOp(np.dot(mat1, mat2))
        self.assertEqual(chan1.compose(chan2, front=True), targ)

        # Compose different dimensions
        chan1 = SuperOp(self.rand_matrix(16, 4), input_dim=2, output_dim=4)
        chan2 = SuperOp(self.rand_matrix(4, 16), output_dim=2)
        chan = chan1.compose(chan2, front=True)
        self.assertEqual(chan.dims, (4, 4))
        chan = chan2.compose(chan1, front=True)
        self.assertEqual(chan.dims, (2, 2))

    def test_compose_front_inplace(self):
        """Test inplace front compose method."""
        # UnitaryChannel evolution
        chan1 = SuperOp(self.sopX)
        chan2 = SuperOp(self.sopY)
        targ = SuperOp(self.sopZ)
        chan1.compose(chan2, inplace=True, front=True)
        self.assertEqual(chan1, targ)

        # 50% depolarizing channel
        chan1 = SuperOp(self.depol_sop(0.5))
        chan1.compose(chan1, inplace=True, front=True)
        targ = SuperOp(self.depol_sop(0.75))
        self.assertEqual(chan1, targ)

        # Random superoperator
        mat1 = self.rand_matrix(4, 4)
        mat2 = self.rand_matrix(4, 4)

        targ = SuperOp(np.dot(mat2, mat1))
        chan1 = SuperOp(mat1)
        chan2 = SuperOp(mat2)
        chan2.compose(chan1, inplace=True, front=True)
        self.assertEqual(chan2, targ)
        targ = SuperOp(np.dot(mat1, mat2))
        chan1 = SuperOp(mat1)
        chan2 = SuperOp(mat2)
        chan1.compose(chan2, inplace=True, front=True)
        self.assertEqual(chan1, targ)

        # Compose different dimensions
        chan1 = SuperOp(self.rand_matrix(16, 4), input_dim=2, output_dim=4)
        chan2 = SuperOp(self.rand_matrix(4, 16), output_dim=2)
        chan1.compose(chan2, inplace=True, front=True)
        self.assertEqual(chan1.dims, (4, 4))
        chan1 = SuperOp(self.rand_matrix(16, 4), input_dim=2, output_dim=4)
        chan2 = SuperOp(self.rand_matrix(4, 16), output_dim=2)
        chan2.compose(chan1, inplace=True, front=True)
        self.assertEqual(chan2.dims, (2, 2))

    def test_expand(self):
        """Test expand method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = np.kron(rho0, rho0)
        chan1 = SuperOp(self.sopI)
        chan2 = SuperOp(self.sopX)

        # X \otimes I
        chan = chan1.expand(chan2)
        rho_targ = np.kron(rho1, rho0)
        self.assertEqual(chan.dims, (4, 4))
        self.assertAllClose(chan._evolve(rho_init), rho_targ)

        # I \otimes X
        chan = chan2.expand(chan1)
        rho_targ = np.kron(rho0, rho1)
        self.assertEqual(chan.dims, (4, 4))
        self.assertAllClose(chan._evolve(rho_init), rho_targ)

    def test_expand_inplace(self):
        """Test inplace expand method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = np.kron(rho0, rho0)

        # X \otimes I
        chan1 = SuperOp(self.sopI)
        chan2 = SuperOp(self.sopX)
        chan1.expand(chan2, inplace=True)
        rho_targ = np.kron(rho1, rho0)
        self.assertEqual(chan1.dims, (4, 4))
        self.assertAllClose(chan1._evolve(rho_init), rho_targ)

        # I \otimes X
        chan1 = SuperOp(self.sopI)
        chan2 = SuperOp(self.sopX)
        chan2.expand(chan1, inplace=True)
        rho_targ = np.kron(rho0, rho1)
        self.assertEqual(chan2.dims, (4, 4))
        self.assertAllClose(chan2._evolve(rho_init), rho_targ)

    def test_tensor(self):
        """Test tensor method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = np.kron(rho0, rho0)
        chan1 = SuperOp(self.sopI)
        chan2 = SuperOp(self.sopX)

        # X \otimes I
        chan = chan2.tensor(chan1)
        rho_targ = np.kron(rho1, rho0)
        self.assertEqual(chan.dims, (4, 4))
        self.assertAllClose(chan._evolve(rho_init), rho_targ)
        chan = chan2 ^ chan1
        self.assertEqual(chan.dims, (4, 4))
        self.assertAllClose(chan._evolve(rho_init), rho_targ)
        # I \otimes X
        chan = chan1.tensor(chan2)
        rho_targ = np.kron(rho0, rho1)
        self.assertEqual(chan.dims, (4, 4))
        self.assertAllClose(chan._evolve(rho_init), rho_targ)
        chan = chan1 ^ chan2
        self.assertEqual(chan.dims, (4, 4))
        self.assertAllClose(chan._evolve(rho_init), rho_targ)

    def test_tensorinplace(self):
        """Test inplace tensor method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = np.kron(rho0, rho0)

        # X \otimes I
        chan1 = SuperOp(self.sopI)
        chan2 = SuperOp(self.sopX)
        chan2.tensor(chan1, inplace=True)
        rho_targ = np.kron(rho1, rho0)
        self.assertEqual(chan2.dims, (4, 4))
        self.assertAllClose(chan2._evolve(rho_init), rho_targ)
        chan1 = SuperOp(self.sopI)
        chan2 = SuperOp(self.sopX)
        chan2 ^= chan1
        self.assertEqual(chan2.dims, (4, 4))
        self.assertAllClose(chan2._evolve(rho_init), rho_targ)

        # I \otimes X
        chan1 = SuperOp(self.sopI)
        chan2 = SuperOp(self.sopX)
        chan1.tensor(chan2, inplace=True)
        rho_targ = np.kron(rho0, rho1)
        self.assertEqual(chan1.dims, (4, 4))
        self.assertAllClose(chan1._evolve(rho_init), rho_targ)
        chan1 = SuperOp(self.sopI)
        chan2 = SuperOp(self.sopX)
        chan1 ^= chan2
        self.assertEqual(chan1.dims, (4, 4))
        self.assertAllClose(chan1._evolve(rho_init), rho_targ)

    def test_power(self):
        """Test power method."""
        # 10% depolarizing channel
        p_id = 0.9
        depol = SuperOp(self.depol_sop(1 - p_id))

        # Compose 3 times
        p_id3 = p_id ** 3
        chan3 = depol.power(3)
        targ3 = SuperOp(self.depol_sop(1 - p_id3))
        self.assertEqual(chan3, targ3)

    def test_power_inplace(self):
        """Test inplace power method."""
        # 10% depolarizing channel
        p_id = 0.9
        depol = SuperOp(self.depol_sop(1 - p_id))

        # Compose 3 times
        p_id3 = p_id ** 3
        depol.power(3, inplace=True)
        targ3 = SuperOp(self.depol_sop(1 - p_id3))
        self.assertEqual(depol, targ3)

    def test_power_except(self):
        """Test power method raises exceptions."""
        chan = SuperOp(self.depol_sop(1))
        # Negative power raises error
        self.assertRaises(QiskitError, chan.power, -1)
        # 0 power raises error
        self.assertRaises(QiskitError, chan.power, 0)
        # Non-integer power raises error
        self.assertRaises(QiskitError, chan.power, 0.5)

    def test_add(self):
        """Test add method."""
        mat1 = 0.5 * self.sopI
        mat2 = 0.5 * self.depol_sop(1)
        targ = SuperOp(mat1 + mat2)

        chan1 = SuperOp(mat1)
        chan2 = SuperOp(mat2)
        self.assertEqual(chan1.add(chan2), targ)
        self.assertEqual(chan1 + chan2, targ)

    def test_add_inplace(self):
        """Test inplace add method."""
        mat1 = 0.5 * self.sopI
        mat2 = 0.5 * self.depol_sop(1)
        targ = SuperOp(mat1 + mat2)

        chan1 = SuperOp(mat1)
        chan2 = SuperOp(mat2)
        chan1.add(chan2, inplace=True)
        self.assertEqual(chan1, targ)

        chan1 = SuperOp(mat1)
        chan2 = SuperOp(mat2)
        chan1 += chan2
        self.assertEqual(chan1, targ)

    def test_add_except(self):
        """Test add method raises exceptions."""
        chan1 = SuperOp(self.sopI)
        chan2 = SuperOp(np.eye(16))
        self.assertRaises(QiskitError, chan1.add, chan2)
        self.assertRaises(QiskitError, chan1.add, 5)

    def test_subtract(self):
        """Test subtract method."""
        mat1 = 0.5 * self.sopI
        mat2 = 0.5 * self.depol_sop(1)
        targ = SuperOp(mat1 - mat2)

        chan1 = SuperOp(mat1)
        chan2 = SuperOp(mat2)
        self.assertEqual(chan1.subtract(chan2), targ)
        self.assertEqual(chan1 - chan2, targ)

    def test_subtract_inplace(self):
        """Test inplace subtract method."""
        mat1 = 0.5 * self.sopI
        mat2 = 0.5 * self.depol_sop(1)
        targ = SuperOp(mat1 - mat2)

        chan1 = SuperOp(mat1)
        chan2 = SuperOp(mat2)
        chan1.subtract(chan2, inplace=True)
        self.assertEqual(chan1, targ)

        chan1 = SuperOp(mat1)
        chan2 = SuperOp(mat2)
        chan1 -= chan2
        self.assertEqual(chan1, targ)

    def test_subtract_except(self):
        """Test subtract method raises exceptions."""
        chan1 = SuperOp(self.sopI)
        chan2 = SuperOp(np.eye(16))
        self.assertRaises(QiskitError, chan1.subtract, chan2)
        self.assertRaises(QiskitError, chan1.subtract, 5)

    def test_multiply(self):
        """Test multiply method."""
        chan = SuperOp(self.sopI)
        val = 0.5
        targ = SuperOp(val * self.sopI)
        self.assertEqual(chan.multiply(val), targ)
        self.assertEqual(val * chan, targ)
        self.assertEqual(chan * val, targ)

    def test_multiply_inplace(self):
        """Test inplace multiply method."""
        chan = SuperOp(self.sopI)
        val = 0.5
        targ = SuperOp(val * self.sopI)
        chan.multiply(val, inplace=True)
        self.assertEqual(chan, targ)

        chan = SuperOp(self.sopI)
        chan *= val
        self.assertEqual(chan, targ)

    def test_multiply_except(self):
        """Test multiply method raises exceptions."""
        chan = SuperOp(self.sopI)
        self.assertRaises(QiskitError, chan.multiply, 's')
        self.assertRaises(QiskitError, chan.multiply, chan)

    def test_negate(self):
        """Test negate method"""
        chan = SuperOp(self.sopI)
        targ = SuperOp(-self.sopI)
        self.assertEqual(-chan, targ)


if __name__ == '__main__':
    unittest.main()
