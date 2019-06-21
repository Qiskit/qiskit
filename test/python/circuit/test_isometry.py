# -*- coding: utf-8 -*-

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
from qiskit.quantum_info.random import random_unitary

from qiskit import BasicAer
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import execute
from qiskit.test import QiskitTestCase
from qiskit.compiler import transpile
from qiskit.quantum_info.operators.predicates import matrix_equal


class TestIsometry(QiskitTestCase):
    """Qiskit isometry tests."""

    def test_isometry(self):
        """Tests for the decomposition of isometries from m to n qubits"""
        for iso in [np.eye(2, 2), random_unitary(2).data, np.eye(4, 4),
                    random_unitary(4).data[:, 0],
                    np.eye(4, 4)[:, 0:2], random_unitary(4).data,
                    np.eye(4, 4)[:, np.random.permutation(np.eye(4, 4).shape[1])][:, 0:2],
                    np.eye(8, 8)[:, np.random.permutation(np.eye(8, 8).shape[1])],
                    random_unitary(8).data[:, 0:4], random_unitary(8).data, random_unitary(16).data,
                    random_unitary(16).data[:, 0:8]]:
            with self.subTest(iso=iso):
                if len(iso.shape) == 1:
                    iso = iso.reshape((len(iso), 1))
                num_q_input = int(np.log2(iso.shape[1]))
                num_q_ancilla_for_output = int(np.log2(iso.shape[0])) - num_q_input
                n = num_q_input + num_q_ancilla_for_output
                q = QuantumRegister(n)
                qc = QuantumCircuit(q)
                qc.iso(iso, q[:num_q_input], q[num_q_input:])
                # Decompose the gate
                qc = transpile(qc, basis_gates=['u1', 'u3', 'u2', 'cx', 'id'])
                # Simulate the decomposed gate
                simulator = BasicAer.get_backend('unitary_simulator')
                result = execute(qc, simulator).result()
                unitary = result.get_unitary(qc)
                iso_from_circuit = unitary[::, 0:2 ** num_q_input]
                iso_desired = iso
                self.assertTrue(matrix_equal(iso_from_circuit, iso_desired, ignore_phase=True))


if __name__ == '__main__':
    unittest.main()
