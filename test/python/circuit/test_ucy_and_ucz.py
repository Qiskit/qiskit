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
from qiskit import execute as q_execute
from qiskit.test import QiskitTestCase

angles_list = [[0], [0.4], [0, 0], [0, 0.8], [0, 0, 1, 1], [0, 1, 0.5, 1],
          (2 * np.pi * np.random.rand(2 ** 3)).tolist(), (2 * np.pi * np.random.rand(2 ** 4)).tolist(),
          (2 * np.pi * np.random.rand(2 ** 5)).tolist()]

rot_axis_list = ["Z", "Y"]


class TestUCY(QiskitTestCase):
    """Qiskit tests for UCY and UCZ rotations gates."""
    def test_ucy(self):
        for angles, rot_axis in itertools.product(angles_list, rot_axis_list):
            with self.subTest(angles=angles, rot_axis=rot_axis):
                num_contr = int(np.log2(len(angles)))
                q = QuantumRegister(num_contr + 1)
                # test the UC R_y gate for all possible basis states.
                for i in range(2 ** (num_contr + 1)):
                    qc = _prepare_basis_state(q, i, num_contr + 1)
                    if rot_axis == "Y":
                        qc.ucy(angles, q[1:num_contr + 1], q[0])
                    else:
                        qc.ucz(angles, q[1:num_contr + 1], q[0])
                    # ToDo: improve efficiency here by allowing to execute circuit on several states in parallel (this would
                    # ToDo: in particular allow to get out the isometry the circuit is implementing by applying it to the first
                    # ToDo: few basis vectors
                    vec_out = np.asarray(q_execute(qc, BasicAer.get_backend(
                        'statevector_simulator')).result().get_statevector(qc, decimals=16))
                    vec_desired = _apply_ucr_to_basis_state(angles, i, rot_axis)
                    # It is fine if the gate is implemented up to a global phase (however, the phases between the different
                    # outputs for different bases states must be correct!)
                    if i == 0:
                        global_phase = _get_global_phase(vec_out, vec_desired)
                    vec_desired = (global_phase * vec_desired).tolist()
                    dist = np.linalg.norm(np.array(vec_desired - vec_out))
                    self.assertAlmostEqual(dist, 0)


def _apply_ucr_to_basis_state(angle_list, basis_state, rot_axis):
    num_qubits = int(np.log2(len(angle_list)) + 1)
    angle = angle_list[basis_state // 2]
    if rot_axis == "Y":
        r = np.array([[np.cos(angle / 2), - np.sin(angle / 2)], [np.sin(angle / 2), np.cos(angle / 2)]])
    else:
        r = np.array([[np.exp(-1.j * angle / 2), 0], [0, np.exp(1.j * angle / 2)]])
    state = np.zeros(2 ** num_qubits, dtype=complex)
    if basis_state / 2. == float(basis_state // 2):
        target_state = np.dot(r, np.array([[1], [0]]))
        state[basis_state] = target_state[0, 0]
        state[basis_state + 1] = target_state[1, 0]
    else:
        target_state = np.dot(r, np.array([[0], [1]]))
        state[basis_state - 1] = target_state[0, 0]
        state[basis_state] = target_state[1, 0]
    return state


def _prepare_basis_state(q, i, num_qubits):
    qc = QuantumCircuit(q)
    # ToDo: Remove this work around after the state vector simulator is fixed (it can't simulate the empty
    # ToDo: circuit at the moment)
    qc.iden(q[0])
    binary_rep = _get_binary_rep_as_list(i, num_qubits)
    for j in range(len(binary_rep)):
        if binary_rep[j] == 1:
            qc.x(q[- (j + 1)])
    return qc


def _get_global_phase(a, b):
    for i in range(len(b)):
        if abs(b[i]) > _EPS:
            return a[i] / b[i]


def _get_binary_rep_as_list(n, num_digits):
    binary_string = np.binary_repr(n).zfill(num_digits)
    binary = []
    for line in binary_string:
        for c in line:
            binary.append(int(c))
    return binary


if __name__ == '__main__':
    unittest.main()
