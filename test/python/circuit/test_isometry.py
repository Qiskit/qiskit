# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Tests for the decomposition of isometries from m to n qubits.
"""

# ToDo: It might be worth to add more functionality to the class QiskitTestCase. In particular, the possibility to check
# ToDo: a gate for a set of possible input vectors (up to a global phase shift). The testing code for UCY,UCZ, UCG and
# ToDo: and SQU could then be simplified.

import unittest

import numpy as np
from scipy.stats import unitary_group

from qiskit import BasicAer
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import execute
from qiskit.test import QiskitTestCase

import math

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class TestUCG(QiskitTestCase):
    """Qiskit isometry tests."""
    def test_isometry(self):
        for iso in [np.eye(4, 4), unitary_group.rvs(4)[:, 0], np.eye(4, 4)[:, 0:2], unitary_group.rvs(4),
                    np.eye(4, 4)[:, np.random.permutation(np.eye(4, 4).shape[1])][:, 0:2],
                    np.eye(8, 8)[:, np.random.permutation(np.eye(8, 8).shape[1])],
                    unitary_group.rvs(8)[:, 0:4], unitary_group.rvs(8), unitary_group.rvs(16),
                    unitary_group.rvs(16)[:, 0:8]]:
            with self.subTest(iso=iso):
                if len(iso.shape) == 1:
                    iso = iso.reshape((len(iso), 1))
                num_q_input = int(np.log2(iso.shape[1]))
                num_q_ancilla_for_output = int(np.log2(iso.shape[0])) - num_q_input
                n = num_q_input + num_q_ancilla_for_output
                q = QuantumRegister(n)
                qc = QuantumCircuit(q)
                qc.iso(iso, q[:num_q_input], q[num_q_input:])
                simulator = BasicAer.get_backend('unitary_simulator')
                result = execute(qc, simulator).result()
                unitary = result.get_unitary(qc)
                iso_from_circuit = unitary[::, 0:2**num_q_input]
                iso_desired = iso
                self.assertTrue(is_identity_up_to_global_phase(np.dot(ct(iso_from_circuit), iso_desired)))


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
