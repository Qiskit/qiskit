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
from qiskit import execute as q_execute
from qiskit.test import QiskitTestCase

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class TestUCG(QiskitTestCase):
    """Qiskit isometry tests."""
    def test_isometry(self):
        for iso in [np.eye(4, 4), unitary_group.rvs(4)[:, 0:2], np.eye(4, 4)[:, 0:2], unitary_group.rvs(4),
                    np.eye(4, 4)[:, np.random.permutation(np.eye(4, 4).shape[1])][:, 0:2],
                    np.eye(8, 8)[:, np.random.permutation(np.eye(8, 8).shape[1])],
                    unitary_group.rvs(8)[:, 0:4], unitary_group.rvs(8), unitary_group.rvs(16),
                    unitary_group.rvs(16)[:, 0:8]]:
            with self.subTest(iso=iso):
                num_q_input = int(np.log2(iso.shape[1]))
                num_q_ancilla_for_output = int(np.log2(iso.shape[0])) - num_q_input
                n = num_q_input + num_q_ancilla_for_output
                q = QuantumRegister(n)
                # test the isometry for all possible input basis states.
                for i in range(2 ** num_q_input):
                    qc = _prepare_basis_state(q, i)
                    qc.iso(iso, q[:num_q_input], q[num_q_input:])
                    # ToDo: improve efficiency here by allowing to execute circuit on several states in parallel (this would
                    # ToDo: in particular allow to get out the isometry the circuit is implementing by applying it to the first
                    # ToDo: few basis vectors)
                    vec_out = np.asarray(q_execute(qc, BasicAer.get_backend(
                        'statevector_simulator')).result().get_statevector(qc, decimals=16))
                    vec_desired = _apply_isometry_to_basis_state(iso, i)
                    # It is fine if the gate is implemented up to a global phase (however, the phases between the different
                    # outputs for different bases states must be correct!
                    if i == 0:
                        global_phase = _get_global_phase(vec_out, vec_desired)
                    vec_desired = (global_phase * vec_desired).tolist()
                    dist = np.linalg.norm(np.array(vec_desired - vec_out))
                    self.assertAlmostEqual(dist, 0)


def _prepare_basis_state(q, i):
    num_qubits = len(q)
    qc = QuantumCircuit(q)
    # ToDo: Remove this work around after the state vector simulator is fixed (it can't simulate the empty
    # ToDo: circuit at the moment)
    qc.iden(q[0])
    binary_rep = _get_binary_rep_as_list(i, num_qubits)
    for j in range(len(binary_rep)):
        if binary_rep[j] == 1:
            qc.x(q[- (j + 1)])
    return qc


def _apply_isometry_to_basis_state(iso, basis_state):
    return iso[:, basis_state]


def _get_binary_rep_as_list(n, num_digits):
    binary_string = np.binary_repr(n).zfill(num_digits)
    binary = []
    for line in binary_string:
        for c in line:
            binary.append(int(c))
    return binary


def _get_global_phase(a, b):
    for i in range(len(b)):
        if abs(b[i]) > _EPS:
            return a[i] / b[i]


if __name__ == '__main__':
    unittest.main()
