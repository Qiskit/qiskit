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

"""Tests for quantum synthesis methods."""

import unittest

import numpy as np
import scipy.linalg as la
from qiskit.extensions.standard import (HGate, IdGate, SdgGate, SGate, U3Gate,
                                        XGate, YGate, ZGate)
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info.synthesis.two_qubit_decompose import euler_angles_1q
from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer
from qiskit.test import QiskitTestCase


def make_oneq_cliffords():
    """Make as list of 1q Cliffords"""
    ixyz_list = [g().to_matrix() for g in (IdGate, XGate, YGate, ZGate)]
    ih_list = [g().to_matrix() for g in (IdGate, HGate)]
    irs_list = [IdGate().to_matrix(),
                SdgGate().to_matrix() @ HGate().to_matrix(),
                HGate().to_matrix() @ SGate().to_matrix()]
    oneq_cliffords = [Operator(ixyz @ ih @ irs) for ixyz in ixyz_list
                      for ih in ih_list
                      for irs in irs_list]
    return oneq_cliffords


ONEQ_CLIFFORDS = make_oneq_cliffords()


def make_hard_thetas_oneq(smallest=1e-18, factor=3.2, steps=22, phi=0.7, lam=0.9):
    """Make 1q gates with theta/2 close to 0, pi/2, pi, 3pi/2"""
    return ([U3Gate(smallest * factor**i, phi, lam) for i in range(steps)] +
            [U3Gate(-smallest * factor**i, phi, lam) for i in range(steps)] +
            [U3Gate(np.pi/2 + smallest * factor**i, phi, lam) for i in range(steps)] +
            [U3Gate(np.pi/2 - smallest * factor**i, phi, lam) for i in range(steps)] +
            [U3Gate(np.pi + smallest * factor**i, phi, lam) for i in range(steps)] +
            [U3Gate(np.pi - smallest * factor**i, phi, lam) for i in range(steps)] +
            [U3Gate(3*np.pi/2 + smallest * factor**i, phi, lam) for i in range(steps)] +
            [U3Gate(3*np.pi/2 - smallest * factor**i, phi, lam) for i in range(steps)])


HARD_THETA_ONEQS = make_hard_thetas_oneq()


# It's too slow to use all 24**4 Clifford combos. If we can make it faster, use a larger set
K1K2S = [(ONEQ_CLIFFORDS[3], ONEQ_CLIFFORDS[5], ONEQ_CLIFFORDS[2], ONEQ_CLIFFORDS[21]),
         (ONEQ_CLIFFORDS[5], ONEQ_CLIFFORDS[6], ONEQ_CLIFFORDS[9], ONEQ_CLIFFORDS[7]),
         (ONEQ_CLIFFORDS[2], ONEQ_CLIFFORDS[1], ONEQ_CLIFFORDS[0], ONEQ_CLIFFORDS[4]),
         [Operator(U3Gate(x, y, z)) for x, y, z in
          [(0.2, 0.3, 0.1), (0.7, 0.15, 0.22), (0.001, 0.97, 2.2), (3.14, 2.1, 0.9)]]]


class TestEulerAngles1Q(QiskitTestCase):
    """Test euler_angles_1q()"""

    def check_one_qubit_euler_angles(self, operator, tolerance=1e-14):
        """Check euler_angles_1q works for the given unitary"""
        with self.subTest(operator=operator):
            target_unitary = operator.data
            angles = euler_angles_1q(target_unitary)
            decomp_unitary = U3Gate(*angles).to_matrix()
            target_unitary *= la.det(target_unitary)**(-0.5)
            decomp_unitary *= la.det(decomp_unitary)**(-0.5)
            maxdist = np.max(np.abs(target_unitary - decomp_unitary))
            if maxdist > 0.1:
                maxdist = np.max(np.abs(target_unitary + decomp_unitary))
            self.assertTrue(np.abs(maxdist) < tolerance, "Worst distance {}".format(maxdist))

    def test_euler_angles_1q_clifford(self):
        """Verify euler_angles_1q produces correct Euler angles for all Cliffords."""
        for clifford in ONEQ_CLIFFORDS:
            self.check_one_qubit_euler_angles(clifford)

    def test_euler_angles_1q_hard_thetas(self):
        """Verify euler_angles_1q for close-to-degenerate theta"""
        for gate in HARD_THETA_ONEQS:
            self.check_one_qubit_euler_angles(Operator(gate))

    def test_euler_angles_1q_random(self, nsamples=100):
        """Verify euler_angles_1q produces correct Euler angles for random unitaries.
        """
        for _ in range(nsamples):
            unitary = random_unitary(2)
            self.check_one_qubit_euler_angles(unitary)


class TestOneQubitEulerDecomposer(QiskitTestCase):
    """Test OneQubitEulerDecomposer"""

    def check_one_qubit_euler_angles(self, operator, basis='U3',
                                     tolerance=1e-12):
        """Check euler_angles_1q works for the given unitary"""
        decomposer = OneQubitEulerDecomposer(basis)
        with self.subTest(operator=operator):
            target_unitary = operator.data
            decomp_unitary = Operator(decomposer(target_unitary)).data
            # Add global phase to make special unitary
            target_unitary *= la.det(target_unitary)**(-0.5)
            decomp_unitary *= la.det(decomp_unitary)**(-0.5)
            maxdist = np.max(np.abs(target_unitary - decomp_unitary))
            if maxdist > 0.1:
                maxdist = np.max(np.abs(target_unitary + decomp_unitary))
            self.assertTrue(np.abs(maxdist) < tolerance,
                            "Worst distance {}".format(maxdist))

    def test_one_qubit_clifford_u3_basis(self):
        """Verify for u3 basis and all Cliffords."""
        for clifford in ONEQ_CLIFFORDS:
            self.check_one_qubit_euler_angles(clifford, 'U3')

    def test_one_qubit_clifford_u1x_basis(self):
        """Verify for u1, x90 basis and all Cliffords."""
        for clifford in ONEQ_CLIFFORDS:
            self.check_one_qubit_euler_angles(clifford, 'U1X')

    def test_one_qubit_clifford_zyz_basis(self):
        """Verify for rz, ry, rz basis and all Cliffords."""
        for clifford in ONEQ_CLIFFORDS:
            self.check_one_qubit_euler_angles(clifford, 'ZYZ')

    def test_one_qubit_clifford_xyx_basis(self):
        """Verify for rx, ry, rx basis and all Cliffords."""
        for clifford in ONEQ_CLIFFORDS:
            self.check_one_qubit_euler_angles(clifford, 'XYX')

    def test_one_qubit_hard_thetas_u3_basis(self):
        """Verify for u3 basis and close-to-degenerate theta."""
        for gate in HARD_THETA_ONEQS:
            self.check_one_qubit_euler_angles(Operator(gate), 'U3')

    def test_one_qubit_hard_thetas_u1x_basis(self):
        """Verify for u1, x90 basis and close-to-degenerate theta."""
        # We lower tolerance for this test since decomposition is
        # less numerically accurate. This is due to it having 5 matrix
        # multiplications and the X90 gates
        for gate in HARD_THETA_ONEQS:
            self.check_one_qubit_euler_angles(Operator(gate), 'U1X', 1e-7)

    def test_one_qubit_hard_thetas_zyz_basis(self):
        """Verify for rz, ry, rz basis and close-to-degenerate theta."""
        for gate in HARD_THETA_ONEQS:
            self.check_one_qubit_euler_angles(Operator(gate), 'ZYZ')

    def test_one_qubit_hard_thetas_xyx_basis(self):
        """Verify for rx, ry, rx basis and close-to-degenerate theta."""
        for gate in HARD_THETA_ONEQS:
            self.check_one_qubit_euler_angles(Operator(gate), 'XYX')

    def test_one_qubit_random_u3_basis(self, nsamples=50):
        """Verify for u3 basis and random unitaries."""
        for _ in range(nsamples):
            unitary = random_unitary(2)
            self.check_one_qubit_euler_angles(unitary, 'U3')

    def test_one_qubit_random_u1x_basis(self, nsamples=50):
        """Verify for u1, x90 basis and random unitaries."""
        for _ in range(nsamples):
            unitary = random_unitary(2)
            self.check_one_qubit_euler_angles(unitary, 'U1X')

    def test_one_qubit_random_zyz_basis(self, nsamples=50):
        """Verify for rz, ry, rz basis and random unitaries."""
        for _ in range(nsamples):
            unitary = random_unitary(2)
            self.check_one_qubit_euler_angles(unitary, 'ZYZ')

    def test_one_qubit_random_xyx_basis(self, nsamples=50):
        """Verify for rx, ry, rx basis and random unitaries."""
        for _ in range(nsamples):
            unitary = random_unitary(2)
            self.check_one_qubit_euler_angles(unitary, 'XYX')


if __name__ == '__main__':
    unittest.main()
