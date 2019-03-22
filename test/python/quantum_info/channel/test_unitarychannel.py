# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring
"""Tests for UnitaryChannel quantum channel representation class."""

import unittest
import numpy as np
import scipy.linalg as la

from qiskit import QiskitError
from qiskit.quantum_info.operators.channel.unitarychannel import UnitaryChannel
from .base import ChannelTestCase


class TestUnitaryChannel(ChannelTestCase):
    """Tests for UnitaryChannel channel representation."""

    def test_init(self):
        """Test initialization"""
        mat = np.eye(3)
        chan = UnitaryChannel(mat)
        self.assertAllClose(chan.data, mat)
        self.assertEqual(chan.dims, (3, 3))

        # Non-square matrix should raise exception
        self.assertRaises(QiskitError, UnitaryChannel, np.zeros((2, 3)))
        # Wrong input or output dims should raise exception
        self.assertRaises(QiskitError, UnitaryChannel, mat, input_dim=2)
        self.assertRaises(QiskitError, UnitaryChannel, mat, output_dim=2)

    def test_equal(self):
        """Test __eq__ method"""
        mat = self.rand_matrix(2, 2)
        self.assertEqual(UnitaryChannel(mat), UnitaryChannel(mat))

    def test_copy(self):
        """Test copy method"""
        mat = np.eye(2)
        orig = UnitaryChannel(mat)
        cpy = orig.copy()
        cpy._data[0, 0] = 0.0
        self.assertFalse(cpy == orig)

    def test_evolve(self):
        """Test evolve method."""
        # Test hadamard
        chan = UnitaryChannel(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
        target_psi = np.array([1, 1]) / np.sqrt(2)
        target_rho = np.array([[0.5, 0.5], [0.5, 0.5]])
        # Test list vector evolve
        self.assertAllClose(chan._evolve([1, 0]), target_psi)
        # Test np.array vector evolve
        self.assertAllClose(chan._evolve(np.array([1, 0])), target_psi)
        # Test list density matrix evolve
        self.assertAllClose(chan._evolve([[1, 0], [0, 0]]), target_rho)
        # Test np.array density matrix evolve
        self.assertAllClose(
            chan._evolve(np.array([[1, 0], [0, 0]])), target_rho)

    def test_is_cptp(self):
        """Test is_cptp method."""
        # X-90 rotation
        X90 = la.expm(-1j * 0.5 * np.pi * np.array([[0, 1], [1, 0]]) / 2)
        self.assertTrue(UnitaryChannel(X90).is_cptp())
        # Non-unitary should return false
        self.assertFalse(UnitaryChannel([[1, 0], [0, 0]]).is_cptp())

    def test_conjugate(self):
        """Test conjugate method."""
        matr = np.array([[1, 2], [3, 4]])
        mati = np.array([[1, 2], [3, 4]])
        chan = UnitaryChannel(matr + 1j * mati)
        uni_conj = chan.conjugate()
        self.assertEqual(uni_conj, UnitaryChannel(matr - 1j * mati))

    def test_conjugate_inplace(self):
        """Test inplace conjugate method."""
        matr = np.array([[1, 2], [3, 4]])
        mati = np.array([[1, 2], [3, 4]])
        chan = UnitaryChannel(matr + 1j * mati)
        chan.conjugate(inplace=True)
        self.assertEqual(chan, UnitaryChannel(matr - 1j * mati))

    def test_transpose(self):
        """Test transpose method."""
        matr = np.array([[1, 2], [3, 4]])
        mati = np.array([[1, 2], [3, 4]])
        chan = UnitaryChannel(matr + 1j * mati)
        uni_t = chan.transpose()
        self.assertEqual(uni_t, UnitaryChannel(matr.T + 1j * mati.T))

    def test_transpose_inplace(self):
        """Test inplace transpose method."""
        matr = np.array([[1, 2], [3, 4]])
        mati = np.array([[1, 2], [3, 4]])
        chan = UnitaryChannel(matr + 1j * mati)
        chan.transpose(inplace=True)
        self.assertEqual(chan, UnitaryChannel(matr.T + 1j * mati.T))

    def test_adjoint(self):
        """Test adjoint method."""
        matr = np.array([[1, 2], [3, 4]])
        mati = np.array([[1, 2], [3, 4]])
        chan = UnitaryChannel(matr + 1j * mati)
        uni_adj = chan.adjoint()
        self.assertEqual(uni_adj, UnitaryChannel(matr.T - 1j * mati.T))

    def test_adjoint_inplace(self):
        """Test inplace adjoint method."""
        matr = np.array([[1, 2], [3, 4]])
        mati = np.array([[1, 2], [3, 4]])
        chan = UnitaryChannel(matr + 1j * mati)
        chan.adjoint(inplace=True)
        self.assertEqual(chan, UnitaryChannel(matr.T - 1j * mati.T))

    def test_compose_except(self):
        """Test compose different dimension exception"""
        self.assertRaises(QiskitError,
                          UnitaryChannel(np.eye(2)).compose,
                          UnitaryChannel(np.eye(3)))
        self.assertRaises(QiskitError,
                          UnitaryChannel(np.eye(2)).compose, np.eye(2))
        self.assertRaises(QiskitError, UnitaryChannel(np.eye(2)).compose, 2)

    def test_compose(self):
        """Test compose method."""
        matX = np.array([[0, 1], [1, 0]], dtype=complex)
        matY = np.array([[0, -1j], [1j, 0]], dtype=complex)

        chan1 = UnitaryChannel(matX)
        chan2 = UnitaryChannel(matY)

        targ = UnitaryChannel(np.dot(matY, matX))
        self.assertEqual(chan1.compose(chan2), targ)
        self.assertEqual(chan1 @ chan2, targ)

        targ = UnitaryChannel(np.dot(matX, matY))
        self.assertEqual(chan2.compose(chan1), targ)
        self.assertEqual(chan2 @ chan1, targ)

    def test_compose_inplace(self):
        """Test inplace compose method."""
        matX = np.array([[0, 1], [1, 0]], dtype=complex)
        matY = np.array([[0, -1j], [1j, 0]], dtype=complex)

        targ = UnitaryChannel(np.dot(matY, matX))
        chan1 = UnitaryChannel(matX)
        chan2 = UnitaryChannel(matY)
        chan1.compose(chan2, inplace=True)
        self.assertEqual(chan1, targ)
        chan1 = UnitaryChannel(matX)
        chan2 = UnitaryChannel(matY)
        chan1 @= chan2
        self.assertEqual(chan1, targ)

        targ = UnitaryChannel(np.dot(matX, matY))
        chan1 = UnitaryChannel(matX)
        chan2 = UnitaryChannel(matY)
        chan2.compose(chan1, inplace=True)
        self.assertEqual(chan2, targ)
        chan1 = UnitaryChannel(matX)
        chan2 = UnitaryChannel(matY)
        chan2 @= chan1
        self.assertEqual(chan2, targ)

    def test_compose_front(self):
        """Test front compose method."""
        matX = np.array([[0, 1], [1, 0]], dtype=complex)
        matY = np.array([[0, -1j], [1j, 0]], dtype=complex)

        chanYX = UnitaryChannel(matY).compose(UnitaryChannel(matX), front=True)
        matYX = np.dot(matY, matX)
        self.assertEqual(chanYX, UnitaryChannel(matYX))

        chanXY = UnitaryChannel(matX).compose(UnitaryChannel(matY), front=True)
        matXY = np.dot(matX, matY)
        self.assertEqual(chanXY, UnitaryChannel(matXY))

    def test_compose_front_inplace(self):
        """Test inplace front compose method."""
        matX = np.array([[0, 1], [1, 0]], dtype=complex)
        matY = np.array([[0, -1j], [1j, 0]], dtype=complex)

        matYX = np.dot(matY, matX)
        chan = UnitaryChannel(matY)
        chan.compose(UnitaryChannel(matX), inplace=True, front=True)
        self.assertEqual(chan, UnitaryChannel(matYX))

        matXY = np.dot(matX, matY)
        chan = UnitaryChannel(matX)
        chan.compose(UnitaryChannel(matY), inplace=True, front=True)
        self.assertEqual(chan, UnitaryChannel(matXY))

    def test_expand(self):
        """Test expand method."""
        mat1 = np.array([[0, 1], [1, 0]], dtype=complex)
        mat2 = np.eye(3, dtype=complex)

        mat21 = np.kron(mat2, mat1)
        chan21 = UnitaryChannel(mat1).expand(UnitaryChannel(mat2))
        self.assertEqual(chan21.dims, (6, 6))
        self.assertEqual(chan21, UnitaryChannel(mat21))

        mat12 = np.kron(mat1, mat2)
        chan12 = UnitaryChannel(mat2).expand(UnitaryChannel(mat1))
        self.assertEqual(chan12.dims, (6, 6))
        self.assertEqual(chan12, UnitaryChannel(mat12))

    def test_expand_inplace(self):
        """Test inplace expand method."""
        mat1 = np.array([[0, 1], [1, 0]], dtype=complex)
        mat2 = np.eye(3, dtype=complex)

        mat21 = np.kron(mat2, mat1)
        chan = UnitaryChannel(mat1)
        chan.expand(UnitaryChannel(mat2), inplace=True)
        self.assertEqual(chan.dims, (6, 6))
        self.assertEqual(chan, UnitaryChannel(mat21))

        mat12 = np.kron(mat1, mat2)
        chan = UnitaryChannel(mat2)
        chan.expand(UnitaryChannel(mat1), inplace=True)
        self.assertEqual(chan.dims, (6, 6))
        self.assertEqual(chan, UnitaryChannel(mat12))

    def test_tensor(self):
        """Test tensor method."""
        mat1 = np.array([[0, 1], [1, 0]], dtype=complex)
        mat2 = np.eye(3, dtype=complex)

        mat21 = np.kron(mat2, mat1)
        chan21 = UnitaryChannel(mat2).tensor(UnitaryChannel(mat1))
        self.assertEqual(chan21.dims, (6, 6))
        self.assertEqual(chan21, UnitaryChannel(mat21))

        mat12 = np.kron(mat1, mat2)
        chan12 = UnitaryChannel(mat1).tensor(UnitaryChannel(mat2))
        self.assertEqual(chan12.dims, (6, 6))
        self.assertEqual(chan12, UnitaryChannel(mat12))

    def test_tensor_inplace(self):
        """Test inplace tensor method."""
        mat1 = np.array([[0, 1], [1, 0]], dtype=complex)
        mat2 = np.eye(3, dtype=complex)

        mat21 = np.kron(mat2, mat1)
        chan = UnitaryChannel(mat2)
        chan.tensor(UnitaryChannel(mat1), inplace=True)
        self.assertEqual(chan.dims, (6, 6))
        self.assertEqual(chan, UnitaryChannel(mat21))

        mat12 = np.kron(mat1, mat2)
        chan = UnitaryChannel(mat1)
        chan.tensor(UnitaryChannel(mat2), inplace=True)
        self.assertEqual(chan.dims, (6, 6))
        self.assertEqual(chan, UnitaryChannel(mat12))

    def test_power(self):
        """Test power method."""
        X90 = la.expm(-1j * 0.5 * np.pi * np.array([[0, 1], [1, 0]]) / 2)
        chan = UnitaryChannel(X90)
        self.assertEqual(chan.power(2), UnitaryChannel([[0, -1j], [-1j, 0]]))
        self.assertEqual(chan.power(4), UnitaryChannel(-1 * np.eye(2)))
        self.assertEqual(chan.power(8), UnitaryChannel(np.eye(2)))

    def test_power_inplace(self):
        """Test inplace power method."""
        X90 = la.expm(-1j * 0.5 * np.pi * np.array([[0, 1], [1, 0]]) / 2)
        chan = UnitaryChannel(X90)
        chan.power(2, inplace=True)
        self.assertEqual(chan, UnitaryChannel([[0, -1j], [-1j, 0]]))
        chan.power(2, inplace=True)
        self.assertEqual(chan, UnitaryChannel(-1 * np.eye(2)))
        chan.power(4, inplace=True)
        self.assertEqual(chan, UnitaryChannel(np.eye(2)))

    def test_power_except(self):
        """Test power method raises exceptions."""
        chan = UnitaryChannel(np.eye(3))
        # Negative power raises error
        self.assertRaises(QiskitError, chan.power, -1)
        # 0 power raises error
        self.assertRaises(QiskitError, chan.power, 0)
        # Non-integer power raises error
        self.assertRaises(QiskitError, chan.power, 0.5)

    def test_add(self):
        """Test add method."""
        chan = UnitaryChannel([[1, 2], [3, 4]])
        self.assertEqual(chan.add(chan), UnitaryChannel([[2, 4], [6, 8]]))
        self.assertEqual(chan + chan, UnitaryChannel([[2, 4], [6, 8]]))

    def test_add_inplace(self):
        """Test inplace add method."""
        chan = UnitaryChannel([[1, 2], [3, 4]])
        chan.add(chan, inplace=True)
        self.assertEqual(chan, UnitaryChannel([[2, 4], [6, 8]]))

        chan = UnitaryChannel([[1, 2], [3, 4]])
        chan += chan
        self.assertEqual(chan, UnitaryChannel([[2, 4], [6, 8]]))

    def test_add_except(self):
        """Test add method raises exceptions."""
        chan1 = UnitaryChannel([[1, 2], [3, 4]])
        chan2 = UnitaryChannel(np.eye(3))
        self.assertRaises(QiskitError, chan1.add, chan2)

    def test_subtract(self):
        """Test subtract method."""
        chan = UnitaryChannel([[1, 2], [3, 4]])
        self.assertEqual(chan.subtract(chan), UnitaryChannel(np.zeros((2, 2))))
        self.assertEqual(chan - chan, UnitaryChannel(np.zeros((2, 2))))

    def test_subtract_inplace(self):
        """Test inplace subtract method."""
        chan = UnitaryChannel([[1, 2], [3, 4]])
        chan.subtract(chan, inplace=True)
        self.assertEqual(chan, UnitaryChannel(np.zeros((2, 2))))

        chan = UnitaryChannel([[1, 2], [3, 4]])
        chan -= chan
        self.assertEqual(chan, UnitaryChannel(np.zeros((2, 2))))

    def test_subtract_except(self):
        """Test subtract method raises exceptions."""
        chan1 = UnitaryChannel([[1, 2], [3, 4]])
        chan2 = UnitaryChannel(np.eye(3))
        self.assertRaises(QiskitError, chan1.subtract, chan2)

    def test_multiply(self):
        """Test multiply method."""
        chan = UnitaryChannel([[1, 2], [3, 4]])
        self.assertEqual(chan.multiply(2), UnitaryChannel([[2, 4], [6, 8]]))
        self.assertEqual(2 * chan, UnitaryChannel([[2, 4], [6, 8]]))

    def test_multiply_inplace(self):
        """Test inplace multiply method."""
        chan = UnitaryChannel([[1, 2], [3, 4]])
        chan.multiply(2.0, inplace=True)
        self.assertEqual(chan, UnitaryChannel([[2, 4], [6, 8]]))

        chan = UnitaryChannel([[1, 2], [3, 4]])
        chan *= 2
        self.assertEqual(chan, UnitaryChannel([[2, 4], [6, 8]]))

    def test_multiply_except(self):
        """Test multiply method raises exceptions."""
        chan = UnitaryChannel([[1, 2], [3, 4]])
        self.assertRaises(QiskitError, chan.multiply, 's')
        self.assertRaises(QiskitError, chan.multiply, chan)

    def test_negate(self):
        """Test negate method"""
        chan = UnitaryChannel([[1, 2], [3, 4]])
        targ = UnitaryChannel([[-1, -2], [-3, -4]])
        self.assertEqual(-chan, targ)


if __name__ == '__main__':
    unittest.main()
