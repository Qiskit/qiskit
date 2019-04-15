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
from qiskit import execute as q_execute
from qiskit.test import QiskitTestCase


class TestDiagGate(QiskitTestCase):
    def test_diag_gate(self):
        for phases in [[0, 0], [0, 0.8], [0, 0, 1, 1], [0, 1, 0.5, 1],
                       (2 * np.pi * np.random.rand(2 ** 3)).tolist(),
                       (2 * np.pi * np.random.rand(2 ** 4)).tolist(), (2 * np.pi * np.random.rand(2 ** 5)).tolist()]:
            with self.subTest(phases=phases):
                diag = [np.exp(1j * ph) for ph in phases]
                num_qubits = int(np.log2(len(diag)))
                q = QuantumRegister(num_qubits)
                # test the diagonal gate for all possible basis states.
                for i in range(2 ** (num_qubits)):
                    qc = _prepare_basis_state(q, i)
                    qc.diag(diag, q[0:num_qubits])
                    vec_out = np.asarray(q_execute(qc, BasicAer.get_backend(
                        'statevector_simulator')).result().get_statevector(qc, decimals=16))
                    vec_desired = _apply_diag_gate_to_basis_state(phases, i)
                    if i == 0:
                        global_phase = vec_out[0] / vec_desired[0]
                    vec_desired = (global_phase * vec_desired).tolist()
                    dist = np.linalg.norm(np.array(vec_desired - vec_out))
                    self.assertAlmostEqual(dist, 0)


def _apply_diag_gate_to_basis_state(phases, basis_state):
    # ToDo: improve efficiency here by implementing a simulation for diagonal gates
    num_qubits = int(np.log2(len(phases)))
    ph = phases[basis_state]
    state = np.zeros(2 ** num_qubits, dtype=complex)
    state[basis_state] = np.exp(1j * ph)
    return state


def _get_binary_rep_as_list(n, num_digits):
    binary_string = np.binary_repr(n).zfill(num_digits)
    binary = []
    for line in binary_string:
        for c in line:
            binary.append(int(c))
    return binary


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


if __name__ == '__main__':
    unittest.main()
