# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring
"""Tests for quantum channel representation class."""

import numpy as np

from qiskit.test import QiskitTestCase


class ChannelTestCase(QiskitTestCase):
    """Tests for Channel representations."""

    # Pauli-matrix unitaries
    matI = np.eye(2)
    matX = np.array([[0, 1], [1, 0]])
    matY = np.array([[0, -1j], [1j, 0]])
    matZ = np.diag([1, -1])
    matH = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    # Pauli-matrix superoperators
    sopI = np.eye(4)
    sopX = np.kron(matX.conj(), matX)
    sopY = np.kron(matY.conj(), matY)
    sopZ = np.kron(matZ.conj(), matZ)
    sopH = np.kron(matH.conj(), matH)

    # Choi-matrices for Pauli-matrix unitaries
    choiI = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
    choiX = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
    choiY = np.array([[0, 0, 0, 0], [0, 1, -1, 0], [0, -1, 1, 0], [0, 0, 0,
                                                                   0]])
    choiZ = np.array([[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0,
                                                                  1]])
    choiH = np.array([[1, 1, 1, -1], [1, 1, 1, -1], [1, 1, 1, -1],
                      [-1, -1, -1, 1]]) / 2

    # Chi-matrices for Pauli-matrix unitaries
    chiI = np.diag([2, 0, 0, 0])
    chiX = np.diag([0, 2, 0, 0])
    chiY = np.diag([0, 0, 2, 0])
    chiZ = np.diag([0, 0, 0, 2])
    chiH = np.array([[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1]])

    # PTM-matrices for Pauli-matrix unitaries
    ptmI = np.diag([1, 1, 1, 1])
    ptmX = np.diag([1, 1, -1, -1])
    ptmY = np.diag([1, -1, 1, -1])
    ptmZ = np.diag([1, -1, -1, 1])
    ptmH = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0]])

    # Depolarizing channels
    def depol_kraus(self, p):
        """Depolarizing channel Kraus operators"""
        return [
            np.sqrt(1 - p * 3 / 4) * self.matI,
            np.sqrt(p / 4) * self.matX,
            np.sqrt(p / 4) * self.matY,
            np.sqrt(p / 4) * self.matZ
        ]

    def depol_sop(self, p):
        """Depolarizing channel superoperator matrix"""
        return (1 - p) * self.sopI + p * np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2

    def depol_choi(self, p):
        """Depolarizing channel Choi-matrix"""
        return (1 - p) * self.choiI + p * np.eye(4) / 2

    def depol_chi(self, p):
        """Depolarizing channel Chi-matrix"""
        return 2 * np.diag([1 - 3 * p / 4, p / 4, p / 4, p / 4])

    def depol_ptm(self, p):
        """Depolarizing channel PTM"""
        return np.diag([1, 1 - p, 1 - p, 1 - p])

    def depol_stine(self, p):
        """Depolarizing channel Stinespring matrix"""
        kraus = self.depol_kraus(p)
        basis = np.eye(4).reshape((4, 4, 1))
        return np.sum([np.kron(k, b) for k, b in zip(kraus, basis)], axis=0)

    def rand_rho(self, n):
        """Return random density matrix"""
        psi = np.random.rand(n) + 1j * np.random.rand(n)
        rho = np.outer(psi, psi.conj())
        rho /= np.trace(rho)
        return rho

    def rand_matrix(self, d1, d2, real=False):
        """Return a random rectangular matrix."""
        if real:
            return np.random.rand(d1, d2)
        return np.random.rand(d1, d2) + 1j * np.random.rand(d1, d2)

    def rand_kraus(self, input_dim, output_dim, n):
        """Return a random (non-CPTP) Kraus operator map"""
        return [self.rand_matrix(output_dim, input_dim) for _ in range(n)]

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
