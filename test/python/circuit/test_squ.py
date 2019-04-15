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
from qiskit import execute as q_execute
from qiskit.test import QiskitTestCase
from qiskit.extensions.quantum_initializer.squ import SingleQubitUnitary


squs = [np.eye(2, 2), np.array([[0., 1.], [1., 0.]]), 1 / np.sqrt(2) * np.array([[1., 1.], [-1., 1.]]),
         np.array([[np.exp(1j * 5. / 2), 0], [0, np.exp(-1j * 5. / 2)]]), unitary_group.rvs(2)]

up_to_diagonal_list = [True, False]

class TestSingleQubitUnitary(QiskitTestCase):
    """Qiskit ZYZ-decomposition tests."""

    def test_squ(self):
        for u, up_to_diagonal in itertools.product(squs, up_to_diagonal_list):
            with self.subTest(u=u,up_to_diagonal=up_to_diagonal):
                q = QuantumRegister(1)
                # test the squ for all possible basis states.
                for i in range(2):
                    qc = _prepare_basis_state(q, i)
                    sqg = SingleQubitUnitary(u, q[0], up_to_diagonal=up_to_diagonal)
                    qc._attach(sqg)
                    # ToDo: improve efficiency here by allowing to execute circuit on several states in parallel (this would
                    # ToDo: in particular allow to get out the isometry the circuit is implementing by applying it to the first
                    # ToDo: few basis vectors
                    vec_out = np.asarray(q_execute(qc, BasicAer.get_backend(
                        'statevector_simulator')).result().get_statevector(qc, decimals=16))
                    if up_to_diagonal:
                        vec_out = np.array(sqg.diag) * vec_out
                    vec_desired = _apply_squ_to_basis_state(u, i)
                    # It is fine if the gate is implemented up to a global phase (however, the phases between the different
                    # outputs for different bases states must be correct!
                    if i == 0:
                        global_phase = _get_global_phase(vec_out, vec_desired)
                    vec_desired = (global_phase * vec_desired).tolist()
                    # Remark: We should not take the fidelity to measure the overlap over the states, since the fidelity ignores
                    # the global phase (and hence the phase relation between the different columns of the unitary that the gate
                    # should implement)
                    dist = np.linalg.norm(np.array(vec_desired - vec_out))
                    self.assertGreater(_EPS, dist)

def _prepare_basis_state(q, i):
    num_qubits=len(q)
    qc = QuantumCircuit(q)
    # ToDo: Remove this work around after the state vector simulator is fixed (it can't simulate the empty
    # ToDo: circuit at the moment)
    qc.iden(q[0])
    binary_rep = _get_binary_rep_as_list(i, num_qubits)
    for j in range(len(binary_rep)):
        if binary_rep[j] == 1:
            qc.x(q[- (j + 1)])
    return qc


def _apply_squ_to_basis_state(squ, basis_state):
    return squ[:, basis_state]


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
