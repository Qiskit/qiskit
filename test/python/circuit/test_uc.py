# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Tests for uniformly controlled single-qubit unitaries.
"""

import unittest

import itertools

import numpy as np
from scipy.linalg import block_diag

from qiskit.extensions.quantum_initializer.uc import UCGate

from qiskit import QuantumCircuit, QuantumRegister, BasicAer, execute
from qiskit.test import QiskitTestCase
from qiskit.quantum_info.random import random_unitary
from qiskit.compiler import transpile
from qiskit.quantum_info.operators.predicates import matrix_equal

_id = np.eye(2, 2)
_not = np.matrix([[0, 1], [1, 0]])

squs_list = [
    [_not],
    [_id],
    [_id, _id],
    [_id, 1j * _id],
    [_id, _not, _id, _not],
    [random_unitary(2).data for i in range(2 ** 2)],
    [random_unitary(2).data for i in range(2 ** 3)],
    [random_unitary(2).data for i in range(2 ** 4)],
]

up_to_diagonal_list = [True, False]


class TestUCGate(QiskitTestCase):
    """Qiskit UCGate tests."""

    def test_ucg(self):
        """Test uniformly controlled gates."""
        for squs, up_to_diagonal in itertools.product(squs_list, up_to_diagonal_list):
            with self.subTest(single_qubit_unitaries=squs, up_to_diagonal=up_to_diagonal):
                num_con = int(np.log2(len(squs)))
                q = QuantumRegister(num_con + 1)
                qc = QuantumCircuit(q)
                qc.uc(squs, q[1:], q[0], up_to_diagonal=up_to_diagonal)
                # Decompose the gate
                qc = transpile(qc, basis_gates=["u1", "u3", "u2", "cx", "id"])
                # Simulate the decomposed gate
                simulator = BasicAer.get_backend("unitary_simulator")
                result = execute(qc, simulator).result()
                unitary = result.get_unitary(qc)
                if up_to_diagonal:
                    ucg = UCGate(squs, up_to_diagonal=up_to_diagonal)
                    unitary = np.dot(np.diagflat(ucg._get_diagonal()), unitary)
                unitary_desired = _get_ucg_matrix(squs)
                self.assertTrue(matrix_equal(unitary_desired, unitary, ignore_phase=True))


def _get_ucg_matrix(squs):
    return block_diag(*squs)


if __name__ == "__main__":
    unittest.main()
