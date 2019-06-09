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
from qiskit import execute
from qiskit.test import QiskitTestCase
from qiskit.compiler import transpile
from qiskit.quantum_info.operators.predicates import matrix_equal


class TestDiagGate(QiskitTestCase):
    def test_diag_gate(self):
        for phases in [[0, 0], [0, 0.8], [0, 0, 1, 1], [0, 1, 0.5, 1],
                       (2 * np.pi * np.random.rand(2 ** 3)).tolist(),
                       (2 * np.pi * np.random.rand(2 ** 4)).tolist(), (2 * np.pi * np.random.rand(2 ** 5)).tolist()]:
            with self.subTest(phases=phases):
                diag = [np.exp(1j * ph) for ph in phases]
                num_qubits = int(np.log2(len(diag)))
                q = QuantumRegister(num_qubits)
                qc = QuantumCircuit(q)
                qc.diag(diag, q[0:num_qubits])
                # Decompose the gate
                qc = transpile(qc, basis_gates=['u1', 'u3', 'u2', 'cx'])
                # Simulate the decomposed gate
                simulator = BasicAer.get_backend('unitary_simulator')
                result = execute(qc, simulator).result()
                unitary = result.get_unitary(qc)
                unitary_desired = _get_diag_gate_matrix(diag)
                self.assertTrue(matrix_equal(unitary, unitary_desired, ignore_phase=True))


def _get_diag_gate_matrix(diag):
    return np.diagflat(diag)

if __name__ == '__main__':
    unittest.main()

