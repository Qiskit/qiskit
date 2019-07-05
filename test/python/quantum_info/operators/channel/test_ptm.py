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

"""Tests for PTM quantum channel representation class."""

import unittest
import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info.operators.channel import PTM
from .channel_test_case import ChannelTestCase


class TestPTM(ChannelTestCase):
    """Tests for PTM channel representation."""

    def test_init(self):
        """Test initialization"""
        mat4 = np.eye(4) / 2.0
        chan = PTM(mat4)
        self.assertAllClose(chan.data, mat4)
        self.assertEqual(chan.dim, (2, 2))

        mat16 = np.eye(16) / 4
        chan = PTM(mat16)
        self.assertAllClose(chan.data, mat16)
        self.assertEqual(chan.dim, (4, 4))

        # Wrong input or output dims should raise exception
        self.assertRaises(QiskitError, PTM, mat16, input_dims=2, output_dims=4)

        # Non multi-qubit dimensions should raise exception
        self.assertRaises(
            QiskitError, PTM, np.eye(6) / 2, input_dims=3, output_dims=2)

    def test_circuit_init(self):
        """Test initialization from a circuit."""
        circuit, target = self.simple_circuit_no_measure()
        op = PTM(circuit)
        target = PTM(target)
        self.assertEqual(op, target)

    def test_circuit_init_except(self):
        """Test initialization from circuit with measure raises exception."""
        circuit = self.simple_circuit_with_measure()
        self.assertRaises(QiskitError, PTM, circuit)

    def test_equal(self):
        """Test __eq__ method"""
        mat = self.rand_matrix(4, 4, real=True)
        self.assertEqual(PTM(mat), PTM(mat))

    def test_copy(self):
        """Test copy method"""
        mat = np.eye(4)
        orig = PTM(mat)
        cpy = orig.copy()
        cpy._data[0, 0] = 0.0
        self.assertFalse(cpy == orig)

    def test_evolve(self):
        """Test evolve method."""
        input_psi = [0, 1]
        input_rho = [[0, 0], [0, 1]]

        # Identity channel
        chan = PTM(self.ptmI)
        target_rho = np.array([[0, 0], [0, 1]])
        self.assertAllClose(chan._evolve(input_psi), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_psi)), target_rho)
        self.assertAllClose(chan._evolve(input_rho), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_rho)), target_rho)

        # Hadamard channel
        chan = PTM(self.ptmH)
        target_rho = np.array([[1, -1], [-1, 1]]) / 2
        self.assertAllClose(chan._evolve(input_psi), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_psi)), target_rho)
        self.assertAllClose(chan._evolve(input_rho), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_rho)), target_rho)

        # Completely depolarizing channel
        chan = PTM(self.depol_ptm(1))
        target_rho = np.eye(2) / 2
        self.assertAllClose(chan._evolve(input_psi), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_psi)), target_rho)
        self.assertAllClose(chan._evolve(input_rho), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_rho)), target_rho)

    def test_is_cptp(self):
        """Test is_cptp method."""
        self.assertTrue(PTM(self.depol_ptm(0.25)).is_cptp())
        # Non-CPTP should return false
        self.assertFalse(
            PTM(1.25 * self.ptmI - 0.25 * self.depol_ptm(1)).is_cptp())

    def test_compose_except(self):
        """Test compose different dimension exception"""
        self.assertRaises(QiskitError, PTM(np.eye(4)).compose, PTM(np.eye(16)))
        self.assertRaises(QiskitError, PTM(np.eye(4)).compose, 2)

    def test_compose(self):
        """Test compose method."""
        # Random input test state
        rho = self.rand_rho(2)

        # UnitaryChannel evolution
        chan1 = PTM(self.ptmX)
        chan2 = PTM(self.ptmY)
        chan = chan1.compose(chan2)
        targ = PTM(self.ptmZ)._evolve(rho)
        self.assertAllClose(chan._evolve(rho), targ)

        # 50% depolarizing channel
        chan1 = PTM(self.depol_ptm(0.5))
        chan = chan1.compose(chan1)
        targ = PTM(self.depol_ptm(0.75))._evolve(rho)
        self.assertAllClose(chan._evolve(rho), targ)

        # Compose random
        ptm1 = self.rand_matrix(4, 4, real=True)
        ptm2 = self.rand_matrix(4, 4, real=True)
        chan1 = PTM(ptm1, input_dims=2, output_dims=2)
        chan2 = PTM(ptm2, input_dims=2, output_dims=2)
        targ = chan2._evolve(chan1._evolve(rho))
        chan = chan1.compose(chan2)
        self.assertEqual(chan.dim, (2, 2))
        self.assertAllClose(chan._evolve(rho), targ)
        chan = chan1 @ chan2
        self.assertEqual(chan.dim, (2, 2))
        self.assertAllClose(chan._evolve(rho), targ)

    def test_compose_front(self):
        """Test front compose method."""
        # Random input test state
        rho = self.rand_rho(2)

        # UnitaryChannel evolution
        chan1 = PTM(self.ptmX)
        chan2 = PTM(self.ptmY)
        chan = chan2.compose(chan1, front=True)
        targ = PTM(self.ptmZ)._evolve(rho)
        self.assertAllClose(chan._evolve(rho), targ)

        # Compose random
        ptm1 = self.rand_matrix(4, 4, real=True)
        ptm2 = self.rand_matrix(4, 4, real=True)
        chan1 = PTM(ptm1, input_dims=2, output_dims=2)
        chan2 = PTM(ptm2, input_dims=2, output_dims=2)
        targ = chan2._evolve(chan1._evolve(rho))
        chan = chan2.compose(chan1, front=True)
        self.assertEqual(chan.dim, (2, 2))
        self.assertAllClose(chan._evolve(rho), targ)

    def test_expand(self):
        """Test expand method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = np.kron(rho0, rho0)
        chan1 = PTM(self.ptmI)
        chan2 = PTM(self.ptmX)

        # X \otimes I
        chan = chan1.expand(chan2)
        rho_targ = np.kron(rho1, rho0)
        self.assertEqual(chan.dim, (4, 4))
        self.assertAllClose(chan._evolve(rho_init), rho_targ)

        # I \otimes X
        chan = chan2.expand(chan1)
        rho_targ = np.kron(rho0, rho1)
        self.assertEqual(chan.dim, (4, 4))
        self.assertAllClose(chan._evolve(rho_init), rho_targ)

        # Completely depolarizing
        chan_dep = PTM(self.depol_ptm(1))
        chan = chan_dep.expand(chan_dep)
        rho_targ = np.diag([1, 1, 1, 1]) / 4
        self.assertEqual(chan.dim, (4, 4))
        self.assertAllClose(chan._evolve(rho_init), rho_targ)

    def test_tensor(self):
        """Test tensor method."""
        rho0, rho1 = np.diag([1, 0]), np.diag([0, 1])
        rho_init = np.kron(rho0, rho0)
        chan1 = PTM(self.ptmI)
        chan2 = PTM(self.ptmX)

        # X \otimes I
        chan = chan2.tensor(chan1)
        rho_targ = np.kron(rho1, rho0)
        self.assertEqual(chan.dim, (4, 4))
        self.assertAllClose(chan._evolve(rho_init), rho_targ)

        # I \otimes X
        chan = chan1.tensor(chan2)
        rho_targ = np.kron(rho0, rho1)
        self.assertEqual(chan.dim, (4, 4))
        self.assertAllClose(chan._evolve(rho_init), rho_targ)

        # Completely depolarizing
        chan_dep = PTM(self.depol_ptm(1))
        chan = chan_dep.tensor(chan_dep)
        rho_targ = np.diag([1, 1, 1, 1]) / 4
        self.assertEqual(chan.dim, (4, 4))
        self.assertAllClose(chan._evolve(rho_init), rho_targ)

    def test_power(self):
        """Test power method."""
        # 10% depolarizing channel
        p_id = 0.9
        depol = PTM(self.depol_ptm(1 - p_id))

        # Compose 3 times
        p_id3 = p_id**3
        chan3 = depol.power(3)
        targ3 = PTM(self.depol_ptm(1 - p_id3))
        self.assertEqual(chan3, targ3)

    def test_power_except(self):
        """Test power method raises exceptions."""
        chan = PTM(self.depol_ptm(1))
        # Non-integer power raises error
        self.assertRaises(QiskitError, chan.power, 0.5)

    def test_add(self):
        """Test add method."""
        mat1 = 0.5 * self.ptmI
        mat2 = 0.5 * self.depol_ptm(1)
        targ = PTM(mat1 + mat2)

        chan1 = PTM(mat1)
        chan2 = PTM(mat2)
        self.assertEqual(chan1.add(chan2), targ)
        self.assertEqual(chan1 + chan2, targ)

    def test_add_except(self):
        """Test add method raises exceptions."""
        chan1 = PTM(self.ptmI)
        chan2 = PTM(np.eye(16))
        self.assertRaises(QiskitError, chan1.add, chan2)
        self.assertRaises(QiskitError, chan1.add, 5)

    def test_subtract(self):
        """Test subtract method."""
        mat1 = 0.5 * self.ptmI
        mat2 = 0.5 * self.depol_ptm(1)
        targ = PTM(mat1 - mat2)

        chan1 = PTM(mat1)
        chan2 = PTM(mat2)
        self.assertEqual(chan1.subtract(chan2), targ)
        self.assertEqual(chan1 - chan2, targ)

    def test_subtract_except(self):
        """Test subtract method raises exceptions."""
        chan1 = PTM(self.ptmI)
        chan2 = PTM(np.eye(16))
        self.assertRaises(QiskitError, chan1.subtract, chan2)
        self.assertRaises(QiskitError, chan1.subtract, 5)

    def test_multiply(self):
        """Test multiply method."""
        chan = PTM(self.ptmI)
        val = 0.5
        targ = PTM(val * self.ptmI)
        self.assertEqual(chan.multiply(val), targ)
        self.assertEqual(val * chan, targ)
        self.assertEqual(chan * val, targ)

    def test_multiply_except(self):
        """Test multiply method raises exceptions."""
        chan = PTM(self.ptmI)
        self.assertRaises(QiskitError, chan.multiply, 's')
        self.assertRaises(QiskitError, chan.multiply, chan)

    def test_negate(self):
        """Test negate method"""
        chan = PTM(self.ptmI)
        targ = PTM(-self.ptmI)
        self.assertEqual(-chan, targ)


if __name__ == '__main__':
    unittest.main()
