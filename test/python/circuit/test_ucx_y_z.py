# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Test the decomposition of uniformly controlled rotations.
"""

_EPS = 1e-10  # global variable used to chop very small numbers to zero

import itertools
import unittest

import numpy as np

from qiskit import BasicAer
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import execute
from qiskit.test import QiskitTestCase
from scipy.linalg import block_diag
import math

angles_list = [[0], [0.4], [0, 0], [0, 0.8], [0, 0, 1, 1], [0, 1, 0.5, 1],
          (2 * np.pi * np.random.rand(2 ** 3)).tolist(), (2 * np.pi * np.random.rand(2 ** 4)).tolist(),
          (2 * np.pi * np.random.rand(2 ** 5)).tolist()]

rot_axis_list = ["X", "Y", "Z"]


class TestUCY(QiskitTestCase):
    """Qiskit tests for UCX, UCY and UCZ rotations gates."""
    def test_ucy(self):
        for angles, rot_axis in itertools.product(angles_list, rot_axis_list):
            with self.subTest(angles=angles, rot_axis=rot_axis):
                num_contr = int(np.log2(len(angles)))
                q = QuantumRegister(num_contr + 1)
                qc = QuantumCircuit(q)
                if rot_axis == "X":
                    qc.ucx(angles, q[1:num_contr + 1], q[0])
                elif rot_axis == "Y":
                    qc.ucy(angles, q[1:num_contr + 1], q[0])
                else:
                    qc.ucz(angles, q[1:num_contr + 1], q[0])
                simulator = BasicAer.get_backend('unitary_simulator')
                result = execute(qc, simulator).result()
                unitary = result.get_unitary(qc)
                unitary_desired = _get_ucr_matrix(angles, rot_axis)
                self.assertTrue(is_identity_up_to_global_phase(np.dot(ct(unitary), unitary_desired)))


def _get_ucr_matrix(angles, rot_axis):
    if rot_axis == "X":
        gates = [np.array([[np.cos(angle / 2), - 1j*np.sin(angle / 2)], [- 1j*np.sin(angle / 2), np.cos(angle / 2)]])
                 for angle in angles]
    elif rot_axis == "Y":
        gates = [np.array([[np.cos(angle / 2), - np.sin(angle / 2)], [np.sin(angle / 2), np.cos(angle / 2)]]) for angle
                 in angles]
    else:
        gates = [np.array([[np.exp(-1.j * angle / 2), 0], [0, np.exp(1.j * angle / 2)]]) for angle in angles]
    return block_diag(*gates)


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
