# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring

"""Tests for Stinespring quantum channel representation class."""

import unittest
import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info.operators.channel import Stinespring
from .base import ChannelTestCase


class TestStinespring(ChannelTestCase):
    """Tests for Stinespring channel representation."""

    def test_init(self):
        """Test initialization"""
        # Initialize from unitary
        chan = Stinespring(self.matI)
        self.assertAllClose(chan.data, self.matI)
        self.assertEqual(chan.dims, (2, 2))

        # Initialize from Stinespring
        chan = Stinespring(self.depol_stine(0.5))
        self.assertAllClose(chan.data, self.depol_stine(0.5))
        self.assertEqual(chan.dims, (2, 2))

        # Initialize from Non-CPTP
        stine_l, stine_r = self.rand_matrix(4, 2), self.rand_matrix(4, 2)
        chan = Stinespring((stine_l, stine_r))
        self.assertAllClose(chan.data, (stine_l, stine_r))
        self.assertEqual(chan.dims, (2, 2))

        # Initialize with redundant second op
        chan = Stinespring((stine_l, stine_l))
        self.assertAllClose(chan.data, stine_l)
        self.assertEqual(chan.dims, (2, 2))

        # Wrong input or output dims should raise exception
        self.assertRaises(QiskitError, Stinespring, stine_l, input_dim=4, output_dim=4)

    def test_equal(self):
        """Test __eq__ method"""
        stine = tuple(self.rand_matrix(4, 2) for _ in range(2))
        self.assertEqual(Stinespring(stine), Stinespring(stine))

    def test_copy(self):
        """Test copy method"""
        mat = np.eye(4)
        orig = Stinespring(mat)
        cpy = orig.copy()
        cpy._data[0][0, 0] = 0.0
        self.assertFalse(cpy == orig)

    def test_evolve(self):
        """Test evolve method."""
        input_psi = [0, 1]
        input_rho = [[0, 0], [0, 1]]

        # Identity channel
        chan = Stinespring(self.matI)
        target_psi = np.array([0, 1])
        self.assertAllClose(chan._evolve(input_psi), target_psi)
        self.assertAllClose(chan._evolve(np.array(input_psi)), target_psi)
        target_rho = np.array([[0, 0], [0, 1]])
        self.assertAllClose(chan._evolve(input_rho), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_rho)), target_rho)

        # Hadamard channel
        mat = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        chan = Stinespring(mat)
        target_psi = np.array([1, -1]) / np.sqrt(2)
        self.assertAllClose(chan._evolve(input_psi), target_psi)
        self.assertAllClose(chan._evolve(np.array(input_psi)), target_psi)
        target_rho = np.array([[1, -1], [-1, 1]]) / 2
        self.assertAllClose(chan._evolve(input_rho), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_rho)), target_rho)

        # Completely depolarizing channel
        chan = Stinespring(self.depol_stine(1))
        target_rho = np.eye(2) / 2
        self.assertAllClose(chan._evolve(input_psi), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_psi)), target_rho)
        self.assertAllClose(chan._evolve(input_rho), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_rho)), target_rho)

    def test_is_cptp(self):
        """Test is_cptp method."""
        self.assertTrue(Stinespring(self.depol_stine(0.5)).is_cptp())
        self.assertTrue(Stinespring(self.matX).is_cptp())
        # Non-CP
        stine_l, stine_r = self.rand_matrix(4, 2), self.rand_matrix(4, 2)
        self.assertFalse(Stinespring((stine_l, stine_r)).is_cptp())
        self.assertFalse(Stinespring(self.matI + self.matX).is_cptp())

    def test_conjugate(self):
        """Test conjugate method."""
        stine_l, stine_r = self.rand_matrix(16, 2), self.rand_matrix(16, 2)
        # Single Stinespring list
        targ = Stinespring(stine_l.conj(), output_dim=4)
        chan1 = Stinespring(stine_l, output_dim=4)
        chan = chan1.conjugate()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dims, (2, 4))
        # Double Stinespring list
        targ = Stinespring((stine_l.conj(), stine_r.conj()), output_dim=4)
        chan1 = Stinespring((stine_l, stine_r), output_dim=4)
        chan = chan1.conjugate()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dims, (2, 4))

    def test_conjugate_inplace(self):
        """Test inplace conjugate method."""
        stine_l, stine_r = self.rand_matrix(16, 2), self.rand_matrix(16, 2)
        # Single Stinespring list
        targ = Stinespring(stine_l.conj(), output_dim=4)
        chan1 = Stinespring(stine_l, output_dim=4)
        chan1.conjugate(inplace=True)
        self.assertEqual(chan1, targ)
        self.assertEqual(chan1.dims, (2, 4))
        # Double Stinespring list
        targ = Stinespring((stine_l.conj(), stine_r.conj()), output_dim=4)
        chan1 = Stinespring((stine_l, stine_r), output_dim=4)
        chan1.conjugate(inplace=True)
        self.assertEqual(chan1, targ)
        self.assertEqual(chan1.dims, (2, 4))

    def test_transpose(self):
        """Test transpose method."""
        stine_l, stine_r = self.rand_matrix(4, 2), self.rand_matrix(4, 2)
        # Single square Stinespring list
        targ = Stinespring(stine_l.T, 4, 2)
        chan1 = Stinespring(stine_l, 2, 4)
        chan = chan1.transpose()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dims, (4, 2))
        # Double square Stinespring list
        targ = Stinespring((stine_l.T, stine_r.T), 4, 2)
        chan1 = Stinespring((stine_l, stine_r), 2, 4)
        chan = chan1.transpose()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dims, (4, 2))

    def test_transpose_inplace(self):
        """Test inplace transpose method."""
        stine_l, stine_r = self.rand_matrix(4, 4), self.rand_matrix(4, 4)
        # Single square Stinespring list
        targ = Stinespring(stine_l.T, 4, 4)
        chan1 = Stinespring(stine_l, 4, 4)
        chan1.transpose(inplace=True)
        self.assertEqual(chan1, targ)
        self.assertEqual(chan1.dims, (4, 4))
        # Double square Stinespring list
        targ = Stinespring((stine_l.T, stine_r.T), 4, 4)
        chan1 = Stinespring((stine_l, stine_r), 4, 4)
        chan1.transpose(inplace=True)
        self.assertEqual(chan1, targ)
        self.assertEqual(chan1.dims, (4, 4))

    def test_adjoint(self):
        """Test adjoint method."""
        stine_l, stine_r = self.rand_matrix(4, 2), self.rand_matrix(4, 2)
        # Single square Stinespring list
        targ = Stinespring(stine_l.T.conj(), 4, 2)
        chan1 = Stinespring(stine_l, 2, 4)
        chan = chan1.adjoint()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dims, (4, 2))
        # Double square Stinespring list
        targ = Stinespring((stine_l.T.conj(), stine_r.T.conj()), 4, 2)
        chan1 = Stinespring((stine_l, stine_r), 2, 4)
        chan = chan1.adjoint()
        self.assertEqual(chan, targ)
        self.assertEqual(chan.dims, (4, 2))

    def test_adjoint_inplace(self):
        """Test inplace adjoint method."""
        stine_l, stine_r = self.rand_matrix(4, 2), self.rand_matrix(4, 2)
        # Single square Stinespring list
        targ = Stinespring(stine_l.T.conj(), 4, 2)
        chan1 = Stinespring(stine_l, 2, 4)
        chan1.adjoint(inplace=True)
        self.assertEqual(chan1, targ)
        self.assertEqual(chan1.dims, (4, 2))
        # Double square Stinespring list
        targ = Stinespring((stine_l.T.conj(), stine_r.T.conj()), 4, 2)
        chan1 = Stinespring((stine_l, stine_r), 2, 4)
        chan1.adjoint(inplace=True)
        self.assertEqual(chan1, targ)
        self.assertEqual(chan1.dims, (4, 2))

    def test_compose_except(self):
        """Test compose different dimension exception"""
        self.assertRaises(QiskitError, Stinespring(np.eye(2)).compose, Stinespring(np.eye(4)))
        self.assertRaises(QiskitError, Stinespring(np.eye(2)).compose, np.eye(2))
        self.assertRaises(QiskitError, Stinespring(np.eye(2)).compose, 2)

    def test_compose(self):
        """Test compose method."""
        # Random input test state
        rho = self.rand_rho(2)

        # UnitaryChannel evolution
        chan1 = Stinespring(self.matX)
        chan2 = Stinespring(self.matY)
        chan = chan1.compose(chan2)
        targ = Stinespring(self.matZ)._evolve(rho)
        self.assertAllClose(chan._evolve(rho), targ)

        # 50% depolarizing channel
        chan1 = Stinespring(self.depol_stine(0.5))
        chan = chan1.compose(chan1)
        targ = Stinespring(self.depol_stine(0.75))._evolve(rho)
        self.assertAllClose(chan._evolve(rho), targ)

        # Compose different dimensions
        stine1, stine2 = self.rand_matrix(16, 2), self.rand_matrix(8, 4)
        chan1 = Stinespring(stine1, input_dim=2, output_dim=4)
        chan2 = Stinespring(stine2, input_dim=4, output_dim=2)
        targ = chan2._evolve(chan1._evolve(rho))
        chan = chan1.compose(chan2)
        self.assertEqual(chan.dims, (2, 2))
        self.assertAllClose(chan._evolve(rho), targ)
        chan = chan1 @ chan2
        self.assertEqual(chan.dims, (2, 2))
        self.assertAllClose(chan._evolve(rho), targ)

    def test_compose_inplace(self):
        """Test inplace compose method."""
        # Random input test state
        rho = self.rand_rho(2)

        # UnitaryChannel evolution
        chan1 = Stinespring(self.matX)
        chan2 = Stinespring(self.matY)
        chan1.compose(chan2, inplace=True)
        targ = Stinespring(self.matZ)._evolve(rho)
        self.assertAllClose(chan1._evolve(rho), targ)

        # 50% depolarizing channel
        chan = Stinespring(self.depol_stine(0.5))
        chan.compose(chan, inplace=True)
        targ = Stinespring(self.depol_stine(0.75))._evolve(rho)
        self.assertAllClose(chan._evolve(rho), targ)

        # Compose different dimensions
        stine1, stine2 = self.rand_matrix(16, 2), self.rand_matrix(8, 4)
        chan1 = Stinespring(stine1, input_dim=2, output_dim=4)
        chan2 = Stinespring(stine2, input_dim=4, output_dim=2)
        targ = chan2._evolve(chan1._evolve(rho))
        chan1.compose(chan2, inplace=True)
        self.assertEqual(chan1.dims, (2, 2))
        self.assertAllClose(chan1._evolve(rho), targ)
        chan1 = Stinespring(stine1, input_dim=2, output_dim=4)
        chan2 = Stinespring(stine2, input_dim=4, output_dim=2)
        chan1 @= chan2
        self.assertEqual(chan1.dims, (2, 2))
        self.assertAllClose(chan1._evolve(rho), targ)

    def test_compose_front(self):
        """Test front compose method."""
        # Random input test state
        rho = self.rand_rho(2)

        # UnitaryChannel evolution
        chan1 = Stinespring(self.matX)
        chan2 = Stinespring(self.matY)
        chan = chan1.compose(chan2, front=True)
        targ = Stinespring(self.matZ)._evolve(rho)
        self.assertAllClose(chan._evolve(rho), targ)

        # 50% depolarizing channel
        chan1 = Stinespring(self.depol_stine(0.5))
        chan = chan1.compose(chan1, front=True)
        targ = Stinespring(self.depol_stine(0.75))._evolve(rho)
        self.assertAllClose(chan._evolve(rho), targ)

        # Compose different dimensions
        stine1, stine2 = self.rand_matrix(16, 2), self.rand_matrix(8, 4)
        chan1 = Stinespring(stine1, input_dim=2, output_dim=4)
        chan2 = Stinespring(stine2, input_dim=4, output_dim=2)
        targ = chan2._evolve(chan1._evolve(rho))
        chan = chan2.compose(chan1, front=True)
        self.assertEqual(chan.dims, (2, 2))
        self.assertAllClose(chan._evolve(rho), targ)

    def test_compose_front_inplace(self):
        """Test inplace front compose method."""
        # Random input test state
        rho = self.rand_rho(2)

        # UnitaryChannel evolution
        chan1 = Stinespring(self.matX)
        chan2 = Stinespring(self.matY)
        chan1.compose(chan2, inplace=True, front=True)
        targ = Stinespring(self.matZ)._evolve(rho)
        self.assertAllClose(chan1._evolve(rho), targ)

        # 50% depolarizing channel
        chan = Stinespring(self.depol_stine(0.5))
        chan.compose(chan, inplace=True, front=True)
        targ = Stinespring(self.depol_stine(0.75))._evolve(rho)
        self.assertAllClose(chan._evolve(rho), targ)

        # Compose different dimensions
        stine1, stine2 = self.rand_matrix(16, 2), self.rand_matrix(8, 4)
        chan1 = Stinespring(stine1, input_dim=2, output_dim=4)
        chan2 = Stinespring(stine2, input_dim=4, output_dim=2)
        targ = chan2._evolve(chan1._evolve(rho))
        chan2.compose(chan1, inplace=True, front=True)
        self.assertEqual(chan2.dims, (2, 2))
        self.assertAllClose(chan2._evolve(rho), targ)

    def test_expand(self):
        """Test expand method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = np.kron(rho0, rho0)
        chan1 = Stinespring(self.matI)
        chan2 = Stinespring(self.matX)

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

        # Completely depolarizing
        chan_dep = Stinespring(self.depol_stine(1))
        chan = chan_dep.expand(chan_dep)
        rho_targ = np.diag([1, 1, 1, 1]) / 4
        self.assertEqual(chan.dims, (4, 4))
        self.assertAllClose(chan._evolve(rho_init), rho_targ)

    def test_expand_inplace(self):
        """Test inplace expand method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = np.kron(rho0, rho0)

        # X \otimes I
        chan1 = Stinespring(self.matI)
        chan2 = Stinespring(self.matX)
        chan1.expand(chan2, inplace=True)
        rho_targ = np.kron(rho1, rho0)
        self.assertEqual(chan1.dims, (4, 4))
        self.assertAllClose(chan1._evolve(rho_init), rho_targ)

        # I \otimes X
        chan1 = Stinespring(self.matI)
        chan2 = Stinespring(self.matX)
        chan2.expand(chan1, inplace=True)
        rho_targ = np.kron(rho0, rho1)
        self.assertEqual(chan2.dims, (4, 4))
        self.assertAllClose(chan2._evolve(rho_init), rho_targ)

        # Completely depolarizing
        chan = Stinespring(self.depol_stine(1))
        chan.expand(chan, inplace=True)
        rho_targ = np.diag([1, 1, 1, 1]) / 4
        self.assertEqual(chan.dims, (4, 4))
        self.assertAllClose(chan._evolve(rho_init), rho_targ)

    def test_tensor(self):
        """Test tensor method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = np.kron(rho0, rho0)
        chan1 = Stinespring(self.matI)
        chan2 = Stinespring(self.matX)

        # X \otimes I
        chan = chan2.tensor(chan1)
        rho_targ = np.kron(rho1, rho0)
        self.assertEqual(chan.dims, (4, 4))
        self.assertAllClose(chan._evolve(rho_init), rho_targ)

        # I \otimes X
        chan = chan1.tensor(chan2)
        rho_targ = np.kron(rho0, rho1)
        self.assertEqual(chan.dims, (4, 4))
        self.assertAllClose(chan._evolve(rho_init), rho_targ)

        # Completely depolarizing
        chan_dep = Stinespring(self.depol_stine(1))
        chan = chan_dep.tensor(chan_dep)
        rho_targ = np.diag([1, 1, 1, 1]) / 4
        self.assertEqual(chan.dims, (4, 4))
        self.assertAllClose(chan._evolve(rho_init), rho_targ)

    def test_tensor_inplace(self):
        """Test inplace tensor method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = np.kron(rho0, rho0)

        # X \otimes I
        chan1 = Stinespring(self.matI)
        chan2 = Stinespring(self.matX)
        chan2.tensor(chan1, inplace=True)
        rho_targ = np.kron(rho1, rho0)
        self.assertEqual(chan2.dims, (4, 4))
        self.assertAllClose(chan2._evolve(rho_init), rho_targ)

        # I \otimes X
        chan1 = Stinespring(self.matI)
        chan2 = Stinespring(self.matX)
        chan1.tensor(chan2, inplace=True)
        rho_targ = np.kron(rho0, rho1)
        self.assertEqual(chan1.dims, (4, 4))
        self.assertAllClose(chan1._evolve(rho_init), rho_targ)

        # Completely depolarizing
        chan = Stinespring(self.depol_stine(1))
        chan.tensor(chan, inplace=True)
        rho_targ = np.diag([1, 1, 1, 1]) / 4
        self.assertEqual(chan.dims, (4, 4))
        self.assertAllClose(chan._evolve(rho_init), rho_targ)

    def test_power(self):
        """Test power method."""
        # 10% depolarizing channel
        rho = np.diag([1, 0])
        p_id = 0.9
        chan = Stinespring(self.depol_stine(1 - p_id))

        # Compose 3 times
        p_id3 = p_id ** 3
        chan3 = chan.power(3)
        targ3a = chan._evolve(chan._evolve(chan._evolve(rho)))
        self.assertAllClose(chan3._evolve(rho), targ3a)
        targ3b = Stinespring(self.depol_stine(1 - p_id3))._evolve(rho)
        self.assertAllClose(chan3._evolve(rho), targ3b)

    def test_power_inplace(self):
        """Test inplace power method."""
        # 10% depolarizing channel
        rho = np.diag([1, 0])
        p_id = 0.9
        chan = Stinespring(self.depol_stine(1 - p_id))

        # Compose 3 times
        p_id3 = p_id ** 3
        targ3a = chan._evolve(chan._evolve(chan._evolve(rho)))
        targ3b = Stinespring(self.depol_stine(1 - p_id3))._evolve(rho)
        chan.power(3, inplace=True)
        self.assertAllClose(chan._evolve(rho), targ3a)
        self.assertAllClose(chan._evolve(rho), targ3b)

    def test_power_except(self):
        """Test power method raises exceptions."""
        chan = Stinespring(self.depol_stine(0.9))
        # Negative power raises error
        self.assertRaises(QiskitError, chan.power, -1)
        # 0 power raises error
        self.assertRaises(QiskitError, chan.power, 0)
        # Non-integer power raises error
        self.assertRaises(QiskitError, chan.power, 0.5)

    def test_add(self):
        """Test add method."""
        # Random input test state
        rho = self.rand_rho(2)
        stine1, stine2 = self.rand_matrix(16, 2), self.rand_matrix(16, 2)

        # Random Single-Stinespring maps
        chan1 = Stinespring(stine1, input_dim=2, output_dim=4)
        chan2 = Stinespring(stine2, input_dim=2, output_dim=4)
        targ = chan1._evolve(rho) + chan2._evolve(rho)
        chan = chan1.add(chan2)
        self.assertAllClose(chan._evolve(rho), targ)
        chan = chan1 + chan2
        self.assertAllClose(chan._evolve(rho), targ)

        # Random Single-Stinespring maps
        chan = Stinespring((stine1, stine2))
        targ = 2 * chan._evolve(rho)
        chan = chan.add(chan)
        self.assertAllClose(chan._evolve(rho), targ)

    def test_add_inplace(self):
        """Test inplace add method."""
        # Random input test state
        rho = self.rand_rho(2)
        stine1, stine2 = self.rand_matrix(16, 2), self.rand_matrix(16, 2)

        # Random Single-Stinespring maps
        chan1 = Stinespring(stine1, input_dim=2, output_dim=4)
        chan2 = Stinespring(stine2, input_dim=2, output_dim=4)
        targ = chan1._evolve(rho) + chan2._evolve(rho)
        chan1.add(chan2, inplace=True)
        self.assertAllClose(chan1._evolve(rho), targ)
        chan1 = Stinespring(stine1, input_dim=2, output_dim=4)
        chan1 += chan2
        self.assertAllClose(chan1._evolve(rho), targ)

        # Random Single-Stinespring maps
        chan = Stinespring((stine1, stine2))
        targ = 2 * chan._evolve(rho)
        chan.add(chan, inplace=True)
        self.assertAllClose(chan._evolve(rho), targ)

    def test_subtract(self):
        """Test subtract method."""
        # Random input test state
        rho = self.rand_rho(2)
        stine1, stine2 = self.rand_matrix(16, 2), self.rand_matrix(16, 2)

        # Random Single-Stinespring maps
        chan1 = Stinespring(stine1, input_dim=2, output_dim=4)
        chan2 = Stinespring(stine2, input_dim=2, output_dim=4)
        targ = chan1._evolve(rho) - chan2._evolve(rho)
        chan = chan1.subtract(chan2)
        self.assertAllClose(chan._evolve(rho), targ)
        chan = chan1 - chan2
        self.assertAllClose(chan._evolve(rho), targ)

        # Random Single-Stinespring maps
        chan = Stinespring((stine1, stine2))
        targ = 0 * chan._evolve(rho)
        chan = chan.subtract(chan)
        self.assertAllClose(chan._evolve(rho), targ)

    def test_subtract_inplace(self):
        """Test inplace subtract method."""
        # Random input test state
        rho = self.rand_rho(2)
        stine1, stine2 = self.rand_matrix(16, 2), self.rand_matrix(16, 2)

        # Random Single-Stinespring maps
        chan1 = Stinespring(stine1, input_dim=2, output_dim=4)
        chan2 = Stinespring(stine2, input_dim=2, output_dim=4)
        targ = chan1._evolve(rho) - chan2._evolve(rho)
        chan1.subtract(chan2, inplace=True)
        self.assertAllClose(chan1._evolve(rho), targ)
        chan1 = Stinespring(stine1, input_dim=2, output_dim=4)
        chan1 -= chan2
        self.assertAllClose(chan1._evolve(rho), targ)

        # Random Single-Stinespring maps
        chan = Stinespring((stine1, stine2))
        targ = 0 * chan._evolve(rho)
        chan.subtract(chan, inplace=True)
        self.assertAllClose(chan._evolve(rho), targ)

    def test_multiply(self):
        """Test multiply method."""
        # Random initial state and Stinespring ops
        rho = self.rand_rho(2)
        val = 0.5
        stine1, stine2 = self.rand_matrix(16, 2), self.rand_matrix(16, 2)

        # Single Stinespring set
        chan1 = Stinespring(stine1, input_dim=2, output_dim=4)
        targ = val * chan1._evolve(rho)
        chan = chan1.multiply(val)
        self.assertAllClose(chan._evolve(rho), targ)
        chan = val * chan1
        self.assertAllClose(chan._evolve(rho), targ)
        chan = chan1 * val
        self.assertAllClose(chan._evolve(rho), targ)

        # Double Stinespring set
        chan2 = Stinespring((stine1, stine2), input_dim=2, output_dim=4)
        targ = val * chan2._evolve(rho)
        chan = chan2.multiply(val)
        self.assertAllClose(chan._evolve(rho), targ)
        chan = val * chan2
        self.assertAllClose(chan._evolve(rho), targ)
        chan = chan2 * val
        self.assertAllClose(chan._evolve(rho), targ)

    def test_multiply_inplace(self):
        """Test inplace multiply method."""
        # Random initial state and Stinespring ops
        rho = self.rand_rho(2)
        val = 0.5
        stine1, stine2 = self.rand_matrix(16, 2), self.rand_matrix(16, 2)

        # Single Stinespring set
        chan1 = Stinespring(stine1, input_dim=2, output_dim=4)
        targ = val * chan1._evolve(rho)
        chan1.multiply(val, inplace=True)
        self.assertAllClose(chan1._evolve(rho), targ)
        chan1 = Stinespring(stine1, input_dim=2, output_dim=4)
        chan1 *= val
        self.assertAllClose(chan1._evolve(rho), targ)

        # Double Stinespring set
        chan2 = Stinespring((stine1, stine2), input_dim=2, output_dim=4)
        targ = val * chan2._evolve(rho)
        chan2.multiply(val, inplace=True)
        self.assertAllClose(chan2._evolve(rho), targ)
        chan2 = Stinespring((stine1, stine2), input_dim=2, output_dim=4)
        chan2 *= val
        self.assertAllClose(chan2._evolve(rho), targ)

    def test_multiply_except(self):
        """Test multiply method raises exceptions."""
        chan = Stinespring(self.depol_stine(1))
        self.assertRaises(QiskitError, chan.multiply, 's')
        self.assertRaises(QiskitError, chan.multiply, chan)

    def test_negate(self):
        """Test negate method"""
        rho = np.diag([1, 0])
        targ = np.diag([-0.5, -0.5])
        chan = -Stinespring(self.depol_stine(1))
        self.assertAllClose(chan._evolve(rho), targ)


if __name__ == '__main__':
    unittest.main()
