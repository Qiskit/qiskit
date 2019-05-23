# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Diagonal gate tests.
"""

import unittest

from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import BasicAer
import numpy as np
from qiskit import execute
from qiskit.test import QiskitTestCase
import math

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class TestDiagGate(QiskitTestCase):
    def test_diag_gate(self):
        for phases in [[0, 0], [0, 0.8], [0, 0, 1, 1], [0, 1, 0.5, 1],
                       (2 * np.pi * np.random.rand(2 ** 3)).tolist(),
                       (2 * np.pi * np.random.rand(2 ** 4)).tolist(), (2 * np.pi * np.random.rand(2 ** 5)).tolist()]:
            with self.subTest(phases=phases):
                diag = [np.exp(1j * ph) for ph in phases]
                num_qubits = int(np.log2(len(diag)))
                q = QuantumRegister(num_qubits)
                qc = QuantumCircuit(q)
                qc.diag(diag, q[0:num_qubits])
                simulator = BasicAer.get_backend('unitary_simulator')
                result = execute(qc, simulator).result()
                unitary = result.get_unitary(qc)
                unitary_desired = _get_diag_gate_matrix(diag)
                self.assertTrue(is_identity_up_to_global_phase(np.dot(unitary,ct(unitary_desired))))


def _get_diag_gate_matrix(diag):
    return np.diagflat(diag)


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
