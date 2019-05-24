# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Tests for uniformly controlled single-qubit unitaries.
"""

import unittest

import itertools

from qiskit.extensions.quantum_initializer.ucg import UCG

from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import BasicAer
import numpy as np
from qiskit import execute
from qiskit.test import QiskitTestCase
from scipy.stats import unitary_group
from scipy.linalg import block_diag
import math

_EPS = 1e-10  # global variable used to chop very small numbers to zero

_id = np.eye(2,2)
_not = np.matrix([[0,1],[1,0]])

squs_list = [[_not],[_id],[_id,_id],[_id,1j*_id],[_id,_not,_id,_not],[unitary_group.rvs(2) for i in range(2**2)],
         [unitary_group.rvs(2) for i in range(2**3)],[unitary_group.rvs(2) for i in range(2**4)]]

up_to_diagonal_list = [True, False]


class TestUCG(QiskitTestCase):
    """Qiskit UCG tests."""
    def test_ucg(self):
        for squs, up_to_diagonal in itertools.product(squs_list, up_to_diagonal_list):
            with self.subTest(single_qubit_unitaries=squs, up_to_diagonal=up_to_diagonal):
                num_con = int(np.log2(len(squs)))
                q = QuantumRegister(num_con + 1)
                qc = QuantumCircuit(q)
                qc.ucg(squs, q[1:], q[0], up_to_diagonal=up_to_diagonal)
                simulator = BasicAer.get_backend('unitary_simulator')
                result = execute(qc, simulator).result()
                unitary = result.get_unitary(qc)
                if up_to_diagonal:
                    ucg = UCG(squs, up_to_diagonal=up_to_diagonal)
                    unitary = np.dot(np.diagflat(ucg.get_diagonal()), unitary)
                unitary_desired = _get_ucg_matrix(squs)
                self.assertTrue(is_identity_up_to_global_phase(np.dot(ct(unitary), unitary_desired)))


def _get_ucg_matrix(squs):
    return block_diag(*squs)


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
