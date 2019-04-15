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
from parameterized import parameterized
from qiskit import execute as q_execute
from qiskit.test import QiskitTestCase
from scipy.stats import unitary_group

_EPS = 1e-10  # global variable used to chop very small numbers to zero

_id = np.eye(2,2)
_not = np.matrix([[0,1],[1,0]])

squs_list = [[_not],[_id,_id],[_id,1j*_id],[_id,_not,_id,_not],[unitary_group.rvs(2) for i in range(2**2)],
         [unitary_group.rvs(2) for i in range(2**3)],[unitary_group.rvs(2) for i in range(2**4)]]

up_to_diagonal_list = [True, False]


class TestUCG(QiskitTestCase):
    """Qiskit UCG tests."""
    def test_ucg(self):
        for squs, up_to_diagonal in itertools.product(squs_list, up_to_diagonal_list):
            with self.subTest(single_qubit_unitaries=squs, up_to_diagonal=up_to_diagonal):
                num_con = int(np.log2(len(squs)))
                q = QuantumRegister(num_con + 1)
                # test the UC gate for all possible basis states.
                for i in range(2 ** (num_con + 1)):
                    qc = _prepare_basis_state(q, i)
                    ucg = UCG(squs, q[1:num_con + 1], q[0], up_to_diagonal=up_to_diagonal)
                    qc._attach(ucg)
                    # ToDo: improve efficiency here by allowing to execute circuit on several states in parallel (this would
                    # ToDo: in particular allow to get out the isometry the circuit is implementing by applying it to the first
                    # ToDo: few basis vectors
                    vec_out = np.asarray(q_execute(qc, BasicAer.get_backend(
                        'statevector_simulator')).result().get_statevector(qc, decimals=16))
                    if up_to_diagonal:
                        vec_out = np.array(ucg.diag) * vec_out
                    vec_desired = _apply_squ_to_basis_state(squs, i)
                    # It is fine if the gate is implemented up to a global phase (however, the phases between the different
                    # outputs for different bases states must be correct!
                    if i == 0:
                        global_phase = _get_global_phase(vec_out, vec_desired)
                    vec_desired = (global_phase * vec_desired).tolist()
                    # Remark: We should not take the fidelity to measure the overlap over the states, since the fidelity ignores
                    # the global phase (and hence the phase relation between the different columns of the unitary that the gate
                    # should implement)
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

def _apply_squ_to_basis_state(squs, basis_state):
    num_qubits = int(np.log2(len(squs)) + 1)
    sqg = squs[basis_state // 2]
    state = np.zeros(2 ** num_qubits, dtype=complex)
    if basis_state/2. == float(basis_state//2):
        target_state = np.dot(sqg, np.array([[1], [0]]))
        state[basis_state] = target_state[0, 0]
        state[basis_state+1] = target_state[1, 0]
    else:
        target_state = np.dot(sqg, np.array([[0], [1]]))
        state[basis_state-1] = target_state[0, 0]
        state[basis_state] = target_state[1, 0]
    return state


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
            return a[i]/b[i]

if __name__ == '__main__':
    unittest.main()
