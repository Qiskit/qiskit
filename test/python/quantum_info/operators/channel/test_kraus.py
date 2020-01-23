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

"""Tests for Kraus quantum channel representation class."""

import unittest
import numpy as np
from numpy.testing import assert_allclose

from qiskit import QiskitError
from qiskit.quantum_info.states import DensityMatrix
from qiskit.quantum_info.operators.channel import Kraus
from .channel_test_case import ChannelTestCase


class TestKraus(ChannelTestCase):
    """Tests for Kraus channel representation."""

    def test_init(self):
        """Test initialization"""
        # Initialize from unitary
        chan = Kraus(self.UI)
        assert_allclose(chan.data, [self.UI])
        self.assertEqual(chan.dim, (2, 2))

        # Initialize from Kraus
        chan = Kraus(self.depol_kraus(0.5))
        assert_allclose(chan.data, self.depol_kraus(0.5))
        self.assertEqual(chan.dim, (2, 2))

        # Initialize from Non-CPTP
        kraus_l, kraus_r = [self.UI, self.UX], [self.UY, self.UZ]
        chan = Kraus((kraus_l, kraus_r))
        assert_allclose(chan.data, (kraus_l, kraus_r))
        self.assertEqual(chan.dim, (2, 2))

        # Initialize with redundant second op
        chan = Kraus((kraus_l, kraus_l))
        assert_allclose(chan.data, kraus_l)
        self.assertEqual(chan.dim, (2, 2))

        # Initialize from rectangular
        kraus = [np.zeros((4, 2))]
        chan = Kraus(kraus)
        assert_allclose(chan.data, kraus)
        self.assertEqual(chan.dim, (2, 4))

        # Wrong input or output dims should raise exception
        self.assertRaises(
            QiskitError, Kraus, kraus, input_dims=4, output_dims=4)

    def test_circuit_init(self):
        """Test initialization from a circuit."""
        circuit, target = self.simple_circuit_no_measure()
        op = Kraus(circuit)
        target = Kraus(target)
        self.assertEqual(op, target)

    def test_circuit_init_except(self):
        """Test initialization from circuit with measure raises exception."""
        circuit = self.simple_circuit_with_measure()
        self.assertRaises(QiskitError, Kraus, circuit)

    def test_equal(self):
        """Test __eq__ method"""
        kraus = [self.rand_matrix(2, 2) for _ in range(2)]
        self.assertEqual(Kraus(kraus), Kraus(kraus))

    def test_copy(self):
        """Test copy method"""
        mat = np.eye(4)
        orig = Kraus(mat)
        cpy = orig.copy()
        cpy._data[0][0][0, 0] = 0.0
        self.assertFalse(cpy == orig)

    def test_is_cptp(self):
        """Test is_cptp method."""
        self.assertTrue(Kraus(self.depol_kraus(0.5)).is_cptp())
        self.assertTrue(Kraus(self.UX).is_cptp())
        # Non-CPTP should return false
        self.assertFalse(Kraus(([self.UI], [self.UX])).is_cptp())
        self.assertFalse(Kraus(([self.UI, self.UX])).is_cptp())

    def test_conjugate(self):
        """Test conjugate method."""
        kraus_l, kraus_r = self.rand_kraus(2, 4, 4), self.rand_kraus(2, 4, 4)
        # Single Kraus list
        targ = Kraus([np.conjugate(k) for k in kraus_l])
        chan1 = Kraus(kraus_l)
        chan = chan1.conjugate()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (2, 4))
        # Double Kraus list
        targ = Kraus(([np.conjugate(k) for k in kraus_l],
                      [np.conjugate(k) for k in kraus_r]))
        chan1 = Kraus((kraus_l, kraus_r))
        chan = chan1.conjugate()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (2, 4))

    def test_transpose(self):
        """Test transpose method."""
        kraus_l, kraus_r = self.rand_kraus(2, 4, 4), self.rand_kraus(2, 4, 4)
        # Single Kraus list
        targ = Kraus([np.transpose(k) for k in kraus_l])
        chan1 = Kraus(kraus_l)
        chan = chan1.transpose()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (4, 2))
        # Double Kraus list
        targ = Kraus(([np.transpose(k) for k in kraus_l],
                      [np.transpose(k) for k in kraus_r]))
        chan1 = Kraus((kraus_l, kraus_r))
        chan = chan1.transpose()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (4, 2))

    def test_adjoint(self):
        """Test adjoint method."""
        kraus_l, kraus_r = self.rand_kraus(2, 4, 4), self.rand_kraus(2, 4, 4)
        # Single Kraus list
        targ = Kraus([np.transpose(k).conj() for k in kraus_l])
        chan1 = Kraus(kraus_l)
        chan = chan1.adjoint()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (4, 2))
        # Double Kraus list
        targ = Kraus(([np.transpose(k).conj() for k in kraus_l],
                      [np.transpose(k).conj() for k in kraus_r]))
        chan1 = Kraus((kraus_l, kraus_r))
        chan = chan1.adjoint()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dim, (4, 2))

    def test_compose_except(self):
        """Test compose different dimension exception"""
        self.assertRaises(QiskitError,
                          Kraus(np.eye(2)).compose, Kraus(np.eye(4)))
        self.assertRaises(QiskitError, Kraus(np.eye(2)).compose, 2)

    def test_compose(self):
        """Test compose method."""
        # Random input test state
        rho = DensityMatrix(self.rand_rho(2))

        # UnitaryChannel evolution
        chan1 = Kraus(self.UX)
        chan2 = Kraus(self.UY)
        chan = chan1.compose(chan2)
        targ = rho @ Kraus(self.UZ)
        self.assertEqual(rho @ chan, targ)

        # 50% depolarizing channel
        chan1 = Kraus(self.depol_kraus(0.5))
        chan = chan1.compose(chan1)
        targ = rho @ Kraus(self.depol_kraus(0.75))
        self.assertEqual(rho @ chan, targ)

        # Compose different dimensions
        kraus1, kraus2 = self.rand_kraus(2, 4, 4), self.rand_kraus(4, 2, 4)
        chan1 = Kraus(kraus1)
        chan2 = Kraus(kraus2)
        targ = rho @ chan1 @ chan2
        chan = chan1.compose(chan2)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(rho @ chan, targ)
        chan = chan1 @ chan2
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(rho @ chan, targ)

    def test_compose_front(self):
        """Test front compose method."""
        # Random input test state
        rho = DensityMatrix(self.rand_rho(2))

        # UnitaryChannel evolution
        chan1 = Kraus(self.UX)
        chan2 = Kraus(self.UY)
        chan = chan1.compose(chan2, front=True)
        targ = rho @ Kraus(self.UZ)
        self.assertEqual(rho @ chan, targ)

        # 50% depolarizing channel
        chan1 = Kraus(self.depol_kraus(0.5))
        chan = chan1.compose(chan1, front=True)
        targ = rho @ Kraus(self.depol_kraus(0.75))
        self.assertEqual(rho @ chan, targ)

        # Compose different dimensions
        kraus1, kraus2 = self.rand_kraus(2, 4, 4), self.rand_kraus(4, 2, 4)
        chan1 = Kraus(kraus1)
        chan2 = Kraus(kraus2)
        targ = rho @ chan1 @ chan2
        chan = chan2.compose(chan1, front=True)
        self.assertEqual(chan.dim, (2, 2))
        self.assertEqual(rho @ chan, targ)

    def test_expand(self):
        """Test expand method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = DensityMatrix(np.kron(rho0, rho0))
        chan1 = Kraus(self.UI)
        chan2 = Kraus(self.UX)

        # X \otimes I
        chan = chan1.expand(chan2)
        rho_targ = DensityMatrix(np.kron(rho1, rho0))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init @ chan, rho_targ)

        # I \otimes X
        chan = chan2.expand(chan1)
        rho_targ = DensityMatrix(np.kron(rho0, rho1))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init @ chan, rho_targ)

        # Completely depolarizing
        chan_dep = Kraus(self.depol_kraus(1))
        chan = chan_dep.expand(chan_dep)
        rho_targ = DensityMatrix(np.diag([1, 1, 1, 1]) / 4)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init @ chan, rho_targ)

    def test_tensor(self):
        """Test tensor method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = DensityMatrix(np.kron(rho0, rho0))
        chan1 = Kraus(self.UI)
        chan2 = Kraus(self.UX)

        # X \otimes I
        chan = chan2.tensor(chan1)
        rho_targ = DensityMatrix(np.kron(rho1, rho0))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init @ chan, rho_targ)

        # I \otimes X
        chan = chan1.tensor(chan2)
        rho_targ = DensityMatrix(np.kron(rho0, rho1))
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init @ chan, rho_targ)

        # Completely depolarizing
        chan_dep = Kraus(self.depol_kraus(1))
        chan = chan_dep.tensor(chan_dep)
        rho_targ = DensityMatrix(np.diag([1, 1, 1, 1]) / 4)
        self.assertEqual(chan.dim, (4, 4))
        self.assertEqual(rho_init @ chan, rho_targ)

    def test_power(self):
        """Test power method."""
        # 10% depolarizing channel
        rho = DensityMatrix(np.diag([1, 0]))
        p_id = 0.9
        chan = Kraus(self.depol_kraus(1 - p_id))

        # Compose 3 times
        p_id3 = p_id**3
        chan3 = chan.power(3)
        targ3a = rho @ chan @ chan @ chan
        self.assertEqual(rho @ chan3, targ3a)
        targ3b = rho @ Kraus(self.depol_kraus(1 - p_id3))
        self.assertEqual(rho @ chan3, targ3b)

    def test_power_except(self):
        """Test power method raises exceptions."""
        chan = Kraus(self.depol_kraus(0.9))
        # Non-integer power raises error
        self.assertRaises(QiskitError, chan.power, 0.5)

    def test_add(self):
        """Test add method."""
        # Random input test state
        rho = DensityMatrix(self.rand_rho(2))
        kraus1, kraus2 = self.rand_kraus(2, 4, 4), self.rand_kraus(2, 4, 4)

        # Random Single-Kraus maps
        chan1 = Kraus(kraus1)
        chan2 = Kraus(kraus2)
        targ = (rho @ chan1) + (rho @ chan2)
        chan = chan1.add(chan2)
        self.assertEqual(rho @ chan, targ)
        chan = chan1 + chan2
        self.assertEqual(rho @ chan, targ)

        # Random Single-Kraus maps
        chan = Kraus((kraus1, kraus2))
        targ = 2 * (rho @ chan)
        chan = chan.add(chan)
        self.assertEqual(rho @ chan, targ)

    def test_subtract(self):
        """Test subtract method."""
        # Random input test state
        rho = DensityMatrix(self.rand_rho(2))
        kraus1, kraus2 = self.rand_kraus(2, 4, 4), self.rand_kraus(2, 4, 4)

        # Random Single-Kraus maps
        chan1 = Kraus(kraus1)
        chan2 = Kraus(kraus2)
        targ = (rho @ chan1) - (rho @ chan2)
        chan = chan1.subtract(chan2)
        self.assertEqual(rho @ chan, targ)
        chan = chan1 - chan2
        self.assertEqual(rho @ chan, targ)

        # Random Single-Kraus maps
        chan = Kraus((kraus1, kraus2))
        targ = 0 * (rho @ chan)
        chan = chan.subtract(chan)
        self.assertEqual(rho @ chan, targ)

    def test_multiply(self):
        """Test multiply method."""
        # Random initial state and Kraus ops
        rho = DensityMatrix(self.rand_rho(2))
        val = 0.5
        kraus1, kraus2 = self.rand_kraus(2, 4, 4), self.rand_kraus(2, 4, 4)

        # Single Kraus set
        chan1 = Kraus(kraus1)
        targ = val * (rho @ chan1)
        chan = chan1.multiply(val)
        self.assertEqual(rho @ chan, targ)
        chan = val * chan1
        self.assertEqual(rho @ chan, targ)
        chan = chan1 * val
        self.assertEqual(rho @ chan, targ)

        # Double Kraus set
        chan2 = Kraus((kraus1, kraus2))
        targ = val * (rho @ chan2)
        chan = chan2.multiply(val)
        self.assertEqual(rho @ chan, targ)
        chan = val * chan2
        self.assertEqual(rho @ chan, targ)
        chan = chan2 * val
        self.assertEqual(rho @ chan, targ)

    def test_multiply_except(self):
        """Test multiply method raises exceptions."""
        chan = Kraus(self.depol_kraus(1))
        self.assertRaises(QiskitError, chan.multiply, 's')
        self.assertRaises(QiskitError, chan.multiply, chan)

    def test_negate(self):
        """Test negate method"""
        rho = DensityMatrix(np.diag([1, 0]))
        targ = DensityMatrix(np.diag([-0.5, -0.5]))
        chan = -Kraus(self.depol_kraus(1))
        self.assertEqual(rho @ chan, targ)


if __name__ == '__main__':
    unittest.main()
