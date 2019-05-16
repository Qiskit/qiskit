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

"""Tests for Chi quantum channel representation class."""

import unittest
import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info.operators.channel import Chi
from .channel_test_case import ChannelTestCase


class TestChi(ChannelTestCase):
    """Tests for Chi channel representation."""

    def test_init(self):
        """Test initialization"""
        mat4 = np.eye(4) / 2.0
        chan = Chi(mat4)
        self.assertAllClose(chan.data, mat4)
        self.assertEqual(chan.dim, (2, 2))

        mat16 = np.eye(16) / 4
        chan = Chi(mat16)
        self.assertAllClose(chan.data, mat16)
        self.assertEqual(chan.dim, (4, 4))

        # Wrong input or output dims should raise exception
        self.assertRaises(QiskitError, Chi, mat16, input_dims=2, output_dims=4)

        # Non multi-qubit dimensions should raise exception
        self.assertRaises(
            QiskitError, Chi, np.eye(6) / 2, input_dims=3, output_dims=2)

    def test_circuit_init(self):
        """Test initialization from a circuit."""
        circuit, target = self.simple_circuit_no_measure()
        op = Chi(circuit)
        target = Chi(target)
        self.assertEqual(op, target)

    def test_circuit_init_except(self):
        """Test initialization from circuit with measure raises exception."""
        circuit = self.simple_circuit_with_measure()
        self.assertRaises(QiskitError, Chi, circuit)

    def test_equal(self):
        """Test __eq__ method"""
        mat = self.rand_matrix(4, 4, real=True)
        self.assertEqual(Chi(mat), Chi(mat))

    def test_copy(self):
        """Test copy method"""
        mat = np.eye(4)
        orig = Chi(mat)
        cpy = orig.copy()
        cpy._data[0, 0] = 0.0
        self.assertFalse(cpy == orig)

    def test_evolve(self):
        """Test evolve method."""
        input_psi = [0, 1]
        input_rho = [[0, 0], [0, 1]]

        # Identity channel
        chan = Chi(self.chiI)
        target_rho = np.array([[0, 0], [0, 1]])
        self.assertAllClose(chan._evolve(input_psi), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_psi)), target_rho)
        self.assertAllClose(chan._evolve(input_rho), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_rho)), target_rho)

        # Hadamard channel
        chan = Chi(self.chiH)
        target_rho = np.array([[1, -1], [-1, 1]]) / 2
        self.assertAllClose(chan._evolve(input_psi), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_psi)), target_rho)
        self.assertAllClose(chan._evolve(input_rho), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_rho)), target_rho)

        # Completely depolarizing channel
        chan = Chi(self.depol_chi(1))
        target_rho = np.eye(2) / 2
        self.assertAllClose(chan._evolve(input_psi), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_psi)), target_rho)
        self.assertAllClose(chan._evolve(input_rho), target_rho)
        self.assertAllClose(chan._evolve(np.array(input_rho)), target_rho)

    def test_is_cptp(self):
        """Test is_cptp method."""
        self.assertTrue(Chi(self.depol_chi(0.25)).is_cptp())
        # Non-CPTP should return false
        self.assertFalse(
            Chi(1.25 * self.chiI - 0.25 * self.depol_chi(1)).is_cptp())

    def test_compose_except(self):
        """Test compose different dimension exception"""
        self.assertRaises(QiskitError, Chi(np.eye(4)).compose, Chi(np.eye(16)))
        self.assertRaises(QiskitError, Chi(np.eye(4)).compose, 2)

    def test_compose(self):
        """Test compose method."""
        # Random input test state
        rho = self.rand_rho(2)

        # UnitaryChannel evolution
        chan1 = Chi(self.chiX)
        chan2 = Chi(self.chiY)
        chan = chan1.compose(chan2)
        targ = Chi(self.chiZ)._evolve(rho)
        self.assertAllClose(chan._evolve(rho), targ)

        # 50% depolarizing channel
        chan1 = Chi(self.depol_chi(0.5))
        chan = chan1.compose(chan1)
        targ = Chi(self.depol_chi(0.75))._evolve(rho)
        self.assertAllClose(chan._evolve(rho), targ)

        # Compose random
        chi1 = self.rand_matrix(4, 4, real=True)
        chi2 = self.rand_matrix(4, 4, real=True)
        chan1 = Chi(chi1, input_dims=2, output_dims=2)
        chan2 = Chi(chi2, input_dims=2, output_dims=2)
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
        chan1 = Chi(self.chiX)
        chan2 = Chi(self.chiY)
        chan = chan2.compose(chan1, front=True)
        targ = Chi(self.chiZ)._evolve(rho)
        self.assertAllClose(chan._evolve(rho), targ)

        # Compose random
        chi1 = self.rand_matrix(4, 4, real=True)
        chi2 = self.rand_matrix(4, 4, real=True)
        chan1 = Chi(chi1, input_dims=2, output_dims=2)
        chan2 = Chi(chi2, input_dims=2, output_dims=2)
        targ = chan2._evolve(chan1._evolve(rho))
        chan = chan2.compose(chan1, front=True)
        self.assertEqual(chan.dim, (2, 2))
        self.assertAllClose(chan._evolve(rho), targ)

    def test_expand(self):
        """Test expand method."""
        # Pauli channels
        paulis = [self.chiI, self.chiX, self.chiY, self.chiZ]
        targs = 4 * np.eye(16)  # diagonals of Pauli channel Chi mats
        for i, chi1 in enumerate(paulis):
            for j, chi2 in enumerate(paulis):
                chan1 = Chi(chi1)
                chan2 = Chi(chi2)
                chan = chan1.expand(chan2)
                # Target for diagonal Pauli channel
                targ = Chi(np.diag(targs[i + 4 * j]))
                self.assertEqual(chan.dim, (4, 4))
                self.assertEqual(chan, targ)

        # Completely depolarizing
        rho = np.diag([1, 0, 0, 0])
        chan_dep = Chi(self.depol_chi(1))
        chan = chan_dep.expand(chan_dep)
        targ = np.diag([1, 1, 1, 1]) / 4
        self.assertEqual(chan.dim, (4, 4))
        self.assertAllClose(chan._evolve(rho), targ)

    def test_tensor(self):
        """Test tensor method."""
        # Pauli channels
        paulis = [self.chiI, self.chiX, self.chiY, self.chiZ]
        targs = 4 * np.eye(16)  # diagonals of Pauli channel Chi mats
        for i, chi1 in enumerate(paulis):
            for j, chi2 in enumerate(paulis):
                chan1 = Chi(chi1)
                chan2 = Chi(chi2)
                chan = chan2.tensor(chan1)
                # Target for diagonal Pauli channel
                targ = Chi(np.diag(targs[i + 4 * j]))
                self.assertEqual(chan.dim, (4, 4))
                self.assertEqual(chan, targ)
                # Test overload
                chan = chan2 ^ chan1
                self.assertEqual(chan.dim, (4, 4))
                self.assertEqual(chan, targ)

        # Completely depolarizing
        rho = np.diag([1, 0, 0, 0])
        chan_dep = Chi(self.depol_chi(1))
        chan = chan_dep.tensor(chan_dep)
        targ = np.diag([1, 1, 1, 1]) / 4
        self.assertEqual(chan.dim, (4, 4))
        self.assertAllClose(chan._evolve(rho), targ)
        # Test operator overload
        chan = chan_dep ^ chan_dep
        self.assertEqual(chan.dim, (4, 4))
        self.assertAllClose(chan._evolve(rho), targ)

    def test_power(self):
        """Test power method."""
        # 10% depolarizing channel
        p_id = 0.9
        depol = Chi(self.depol_chi(1 - p_id))

        # Compose 3 times
        p_id3 = p_id**3
        chan3 = depol.power(3)
        targ3 = Chi(self.depol_chi(1 - p_id3))
        self.assertEqual(chan3, targ3)

    def test_power_except(self):
        """Test power method raises exceptions."""
        chan = Chi(self.depol_chi(1))
        # Non-integer power raises error
        self.assertRaises(QiskitError, chan.power, 0.5)

    def test_add(self):
        """Test add method."""
        mat1 = 0.5 * self.chiI
        mat2 = 0.5 * self.depol_chi(1)
        targ = Chi(mat1 + mat2)

        chan1 = Chi(mat1)
        chan2 = Chi(mat2)
        self.assertEqual(chan1.add(chan2), targ)
        self.assertEqual(chan1 + chan2, targ)

    def test_add_except(self):
        """Test add method raises exceptions."""
        chan1 = Chi(self.chiI)
        chan2 = Chi(np.eye(16))
        self.assertRaises(QiskitError, chan1.add, chan2)
        self.assertRaises(QiskitError, chan1.add, 5)

    def test_subtract(self):
        """Test subtract method."""
        mat1 = 0.5 * self.chiI
        mat2 = 0.5 * self.depol_chi(1)
        targ = Chi(mat1 - mat2)

        chan1 = Chi(mat1)
        chan2 = Chi(mat2)
        self.assertEqual(chan1.subtract(chan2), targ)
        self.assertEqual(chan1 - chan2, targ)

    def test_subtract_except(self):
        """Test subtract method raises exceptions."""
        chan1 = Chi(self.chiI)
        chan2 = Chi(np.eye(16))
        self.assertRaises(QiskitError, chan1.subtract, chan2)
        self.assertRaises(QiskitError, chan1.subtract, 5)

    def test_multiply(self):
        """Test multiply method."""
        chan = Chi(self.chiI)
        val = 0.5
        targ = Chi(val * self.chiI)
        self.assertEqual(chan.multiply(val), targ)
        self.assertEqual(val * chan, targ)
        self.assertEqual(chan * val, targ)

    def test_multiply_except(self):
        """Test multiply method raises exceptions."""
        chan = Chi(self.chiI)
        self.assertRaises(QiskitError, chan.multiply, 's')
        self.assertRaises(QiskitError, chan.multiply, chan)

    def test_negate(self):
        """Test negate method"""
        chan = Chi(self.chiI)
        targ = Chi(-self.chiI)
        self.assertEqual(-chan, targ)


if __name__ == '__main__':
    unittest.main()
