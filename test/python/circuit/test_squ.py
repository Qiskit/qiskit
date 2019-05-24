# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
import itertools


_EPS = 1e-10  # global variable used to chop very small numbers to zero

"""
Tests for the ZYZ decomposition for single-qubit unitary.
"""

import unittest

import numpy as np
from scipy.stats import unitary_group

from qiskit import BasicAer
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import execute
from qiskit.test import QiskitTestCase
from qiskit.extensions.quantum_initializer.squ import SingleQubitUnitary
import math


squs = [np.eye(2, 2), np.array([[0., 1.], [1., 0.]]), 1 / np.sqrt(2) * np.array([[1., 1.], [-1., 1.]]),
         np.array([[np.exp(1j * 5. / 2), 0], [0, np.exp(-1j * 5. / 2)]]), unitary_group.rvs(2)]

up_to_diagonal_list = [True, False]


class TestSingleQubitUnitary(QiskitTestCase):
    """Qiskit ZYZ-decomposition tests."""

    def test_squ(self):
        for u, up_to_diagonal in itertools.product(squs, up_to_diagonal_list):
            with self.subTest(u=u,up_to_diagonal=up_to_diagonal):
                qr = QuantumRegister(1, "qr")
                qc = QuantumCircuit(qr)
                qc.squ(u, qr[0], up_to_diagonal=up_to_diagonal)
                simulator = BasicAer.get_backend('unitary_simulator')
                result = execute(qc, simulator).result()
                unitary = result.get_unitary(qc)
                if up_to_diagonal:
                    squ = SingleQubitUnitary(u, up_to_diagonal=up_to_diagonal)
                    unitary = np.dot(np.diagflat(squ.get_diag()), unitary)
                unitary_desired = u
                self.assertTrue(is_identity_up_to_global_phase(np.dot(ct(unitary), unitary_desired)))


def is_identity_up_to_global_phase(m):
    if not abs(abs(m[0, 0])-1) < _EPS:
        return False
    phase = m[0, 0]
    err = np.linalg.norm(1/phase * m - np.eye(m.shape[1], m.shape[1]))
    return math.isclose(err, 0, abs_tol=_EPS)


def ct(m):
    return np.transpose(np.conjugate(m))


if __name__ == '__main__':
    unittest.main()
