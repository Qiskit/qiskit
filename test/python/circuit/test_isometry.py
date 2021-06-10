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

"""Isometry tests."""

import unittest

import numpy as np
from ddt import ddt, data

from qiskit.quantum_info.random import random_unitary
from qiskit import BasicAer
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import execute
from qiskit.test import QiskitTestCase
from qiskit.compiler import transpile
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info import Operator
from qiskit.extensions.quantum_initializer.isometry import Isometry


@ddt
class TestIsometry(QiskitTestCase):
    """Qiskit isometry tests."""

    @data(
        np.eye(2, 2),
        random_unitary(2).data,
        np.eye(4, 4),
        random_unitary(4).data[:, 0],
        np.eye(4, 4)[:, 0:2],
        random_unitary(4).data,
        np.eye(4, 4)[:, np.random.permutation(np.eye(4, 4).shape[1])][:, 0:2],
        np.eye(8, 8)[:, np.random.permutation(np.eye(8, 8).shape[1])],
        random_unitary(8).data[:, 0:4],
        random_unitary(8).data,
        random_unitary(16).data,
        random_unitary(16).data[:, 0:8],
    )
    def test_isometry(self, iso):
        """Tests for the decomposition of isometries from m to n qubits"""
        if len(iso.shape) == 1:
            iso = iso.reshape((len(iso), 1))
        num_q_output = int(np.log2(iso.shape[0]))
        num_q_input = int(np.log2(iso.shape[1]))
        q = QuantumRegister(num_q_output)
        qc = QuantumCircuit(q)
        qc.isometry(iso, q[:num_q_input], q[num_q_input:])

        # Verify the circuit can be decomposed
        self.assertIsInstance(qc.decompose(), QuantumCircuit)

        # Decompose the gate
        qc = transpile(qc, basis_gates=["u1", "u3", "u2", "cx", "id"])

        # Simulate the decomposed gate
        simulator = BasicAer.get_backend("unitary_simulator")
        result = execute(qc, simulator).result()
        unitary = result.get_unitary(qc)
        iso_from_circuit = unitary[::, 0 : 2 ** num_q_input]
        iso_desired = iso
        self.assertTrue(matrix_equal(iso_from_circuit, iso_desired, ignore_phase=True))

    @data(
        np.eye(2, 2),
        random_unitary(2).data,
        np.eye(4, 4),
        random_unitary(4).data[:, 0],
        np.eye(4, 4)[:, 0:2],
        random_unitary(4).data,
        np.eye(4, 4)[:, np.random.permutation(np.eye(4, 4).shape[1])][:, 0:2],
        np.eye(8, 8)[:, np.random.permutation(np.eye(8, 8).shape[1])],
        random_unitary(8).data[:, 0:4],
        random_unitary(8).data,
        random_unitary(16).data,
        random_unitary(16).data[:, 0:8],
    )
    def test_isometry_tolerance(self, iso):
        """Tests for the decomposition of isometries from m to n qubits with a custom tolerance"""
        if len(iso.shape) == 1:
            iso = iso.reshape((len(iso), 1))
        num_q_output = int(np.log2(iso.shape[0]))
        num_q_input = int(np.log2(iso.shape[1]))
        q = QuantumRegister(num_q_output)
        qc = QuantumCircuit(q)

        # Compute isometry with custom tolerance
        qc.isometry(iso, q[:num_q_input], q[num_q_input:], epsilon=1e-3)

        # Verify the circuit can be decomposed
        self.assertIsInstance(qc.decompose(), QuantumCircuit)

        # Decompose the gate
        qc = transpile(qc, basis_gates=["u1", "u3", "u2", "cx", "id"])

        # Simulate the decomposed gate
        simulator = BasicAer.get_backend("unitary_simulator")
        result = execute(qc, simulator).result()
        unitary = result.get_unitary(qc)
        iso_from_circuit = unitary[::, 0 : 2 ** num_q_input]
        self.assertTrue(matrix_equal(iso_from_circuit, iso, ignore_phase=True))

    @data(
        np.eye(2, 2),
        random_unitary(2).data,
        np.eye(4, 4),
        random_unitary(4).data[:, 0],
        np.eye(4, 4)[:, 0:2],
        random_unitary(4).data,
        np.eye(4, 4)[:, np.random.permutation(np.eye(4, 4).shape[1])][:, 0:2],
        np.eye(8, 8)[:, np.random.permutation(np.eye(8, 8).shape[1])],
        random_unitary(8).data[:, 0:4],
        random_unitary(8).data,
        random_unitary(16).data,
        random_unitary(16).data[:, 0:8],
    )
    def test_isometry_inverse(self, iso):
        """Tests for the inverse of isometries from m to n qubits"""
        if len(iso.shape) == 1:
            iso = iso.reshape((len(iso), 1))

        num_q_output = int(np.log2(iso.shape[0]))

        q = QuantumRegister(num_q_output)
        qc = QuantumCircuit(q)
        qc.append(Isometry(iso, 0, 0), q)
        qc.append(Isometry(iso, 0, 0).inverse(), q)

        result = Operator(qc)
        np.testing.assert_array_almost_equal(result.data, np.identity(result.dim[0]))


if __name__ == "__main__":
    unittest.main()
