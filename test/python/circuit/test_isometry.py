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
from qiskit.quantum_info import Operator
from qiskit.circuit.library.generalized_gates import Isometry


@ddt
class TestIsometry(QiskitTestCase):
    """Qiskit isometry tests."""

    @data(
        np.eye(2, 2),
        random_unitary(2, seed=868540).data,
        np.eye(4, 4),
        random_unitary(4, seed=16785).data[:, 0],
        np.eye(4, 4)[:, 0:2],
        random_unitary(4, seed=660477).data,
        np.eye(4, 4)[:, np.random.RandomState(seed=719010).permutation(4)][:, 0:2],
        np.eye(8, 8)[:, np.random.RandomState(seed=544326).permutation(8)],
        random_unitary(8, seed=247924).data[:, 0:4],
        random_unitary(8, seed=765720).data,
        random_unitary(16, seed=278663).data,
        random_unitary(16, seed=406498).data[:, 0:8],
    )
    def test_isometry(self, iso):
        """Tests for the decomposition of isometries from m to n qubits"""
        if len(iso.shape) == 1:
            iso = iso.reshape((len(iso), 1))
        num_q_output = int(np.log2(iso.shape[0]))
        num_q_input = int(np.log2(iso.shape[1]))
        q = QuantumRegister(num_q_output)
        qc = QuantumCircuit(q)

        with self.assertWarns(PendingDeprecationWarning):
            qc.iso(iso, q[:num_q_input], q[num_q_input:])

        # Verify the circuit can be decomposed
        self.assertIsInstance(qc.decompose(), QuantumCircuit)

        # Decompose the gate
        qc = transpile(qc, basis_gates=["u1", "u3", "u2", "cx", "id"])

        # Simulate the decomposed gate
        simulator = BasicAer.get_backend("unitary_simulator")
        result = execute(qc, simulator).result()
        unitary = result.get_unitary(qc)
        iso_from_circuit = unitary[::, 0 : 2**num_q_input]
        iso_desired = iso

        self.assertTrue(np.allclose(iso_from_circuit, iso_desired))

    @data(
        np.eye(2, 2),
        random_unitary(2, seed=99506).data,
        np.eye(4, 4),
        random_unitary(4, seed=673459).data[:, 0],
        np.eye(4, 4)[:, 0:2],
        random_unitary(4, seed=124090).data,
        np.eye(4, 4)[:, np.random.RandomState(seed=889848).permutation(4)][:, 0:2],
        np.eye(8, 8)[:, np.random.RandomState(seed=94795).permutation(8)],
        random_unitary(8, seed=986292).data[:, 0:4],
        random_unitary(8, seed=632121).data,
        random_unitary(16, seed=623107).data,
        random_unitary(16, seed=889326).data[:, 0:8],
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
        with self.assertWarns(PendingDeprecationWarning):
            qc.isometry(iso, q[:num_q_input], q[num_q_input:], epsilon=1e-3)

        # Verify the circuit can be decomposed
        self.assertIsInstance(qc.decompose(), QuantumCircuit)

        # Decompose the gate
        qc = transpile(qc, basis_gates=["u1", "u3", "u2", "cx", "id"])

        # Simulate the decomposed gate
        simulator = BasicAer.get_backend("unitary_simulator")
        result = execute(qc, simulator).result()
        unitary = result.get_unitary(qc)
        iso_from_circuit = unitary[::, 0 : 2**num_q_input]

        self.assertTrue(np.allclose(iso_from_circuit, iso))

    @data(
        np.eye(2, 2),
        random_unitary(2, seed=272225).data,
        np.eye(4, 4),
        random_unitary(4, seed=592640).data[:, 0],
        np.eye(4, 4)[:, 0:2],
        random_unitary(4, seed=714210).data,
        np.eye(4, 4)[:, np.random.RandomState(seed=719934).permutation(4)][:, 0:2],
        np.eye(8, 8)[:, np.random.RandomState(seed=284469).permutation(8)],
        random_unitary(8, seed=656745).data[:, 0:4],
        random_unitary(8, seed=583813).data,
        random_unitary(16, seed=101363).data,
        random_unitary(16, seed=583429).data[:, 0:8],
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
