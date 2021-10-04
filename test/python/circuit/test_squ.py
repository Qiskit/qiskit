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

# pylint: disable=invalid-name

"""Single-qubit unitary tests."""

import itertools
import unittest
import numpy as np
from qiskit.quantum_info.random import random_unitary
from qiskit import BasicAer
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import execute
from qiskit.test import QiskitTestCase
from qiskit.extensions.quantum_initializer.squ import SingleQubitUnitary
from qiskit.compiler import transpile
from qiskit.quantum_info.operators.predicates import matrix_equal

squs = [
    np.eye(2, 2),
    np.array([[0.0, 1.0], [1.0, 0.0]]),
    1 / np.sqrt(2) * np.array([[1.0, 1.0], [-1.0, 1.0]]),
    np.array([[np.exp(1j * 5.0 / 2), 0], [0, np.exp(-1j * 5.0 / 2)]]),
    random_unitary(2).data,
]

up_to_diagonal_list = [True, False]


class TestSingleQubitUnitary(QiskitTestCase):
    """Qiskit ZYZ-decomposition tests."""

    def test_squ(self):
        """Tests for single-qubit unitary decomposition."""
        for u, up_to_diagonal in itertools.product(squs, up_to_diagonal_list):
            with self.subTest(u=u, up_to_diagonal=up_to_diagonal):
                qr = QuantumRegister(1, "qr")
                qc = QuantumCircuit(qr)
                qc.squ(u, qr[0], up_to_diagonal=up_to_diagonal)
                # Decompose the gate
                qc = transpile(qc, basis_gates=["u1", "u3", "u2", "cx", "id"])
                # Simulate the decomposed gate
                simulator = BasicAer.get_backend("unitary_simulator")
                result = execute(qc, simulator).result()
                unitary = result.get_unitary(qc)
                if up_to_diagonal:
                    squ = SingleQubitUnitary(u, up_to_diagonal=up_to_diagonal)
                    unitary = np.dot(np.diagflat(squ.diag), unitary)
                unitary_desired = u
                self.assertTrue(matrix_equal(unitary_desired, unitary, ignore_phase=True))


if __name__ == "__main__":
    unittest.main()
