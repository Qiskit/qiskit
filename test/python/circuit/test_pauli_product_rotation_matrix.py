# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0.
import unittest

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.generalized_gates.pauli_product_rotation import (
    PauliProductRotationGate,
)
from qiskit.quantum_info import Operator, Pauli, Statevector


class TestPauliProductRotationMatrix(unittest.TestCase):
    def test_single_qubit_x_rotation(self):
        theta = np.pi / 3
        gate = PauliProductRotationGate(Pauli("X"), theta)
        mat = np.asarray(Operator(gate).data)
        self.assertEqual(mat.shape, (2, 2))
        self.assertTrue(np.allclose(mat.conj().T @ mat, np.eye(2)))

    def test_single_qubit_z_rotation(self):
        theta = np.pi / 4
        gate = PauliProductRotationGate(Pauli("Z"), theta)
        mat = np.asarray(Operator(gate).data)
        self.assertEqual(mat.shape, (2, 2))
        self.assertTrue(np.allclose(mat.conj().T @ mat, np.eye(2)))

    def test_two_qubit_zz(self):
        theta = np.pi / 5
        gate = PauliProductRotationGate(Pauli("ZZ"), theta)
        mat = np.asarray(Operator(gate).data)
        self.assertEqual(mat.shape, (4, 4))
        self.assertTrue(np.allclose(mat.conj().T @ mat, np.eye(4)))

    def test_three_qubit_xyz(self):
        theta = 0.7
        gate = PauliProductRotationGate(Pauli("XYZ"), theta)
        mat = np.asarray(Operator(gate).data)
        self.assertEqual(mat.shape, (8, 8))
        self.assertTrue(np.allclose(mat.conj().T @ mat, np.eye(8)))

    def test_identity_pauli_is_global_phase(self):
        theta = np.pi / 6
        gate = PauliProductRotationGate(Pauli("II"), theta)
        mat = np.asarray(Operator(gate).data)
        expected = np.exp(-1j * theta / 2) * np.eye(4)
        self.assertTrue(np.allclose(mat, expected))

    def test_theta_zero_is_identity(self):
        gate = PauliProductRotationGate(Pauli("XZ"), 0.0)
        mat = np.asarray(Operator(gate).data)
        self.assertTrue(np.allclose(mat, np.eye(4)))

    def test_theta_2pi_is_negative_identity(self):
        gate = PauliProductRotationGate(Pauli("Z"), 2 * np.pi)
        mat = np.asarray(Operator(gate).data)
        self.assertTrue(np.allclose(mat, -np.eye(2)))

    def test_matrix_consistent_with_simulation(self):
        theta = np.pi / 3
        gate = PauliProductRotationGate(Pauli("ZZ"), theta)

        qc = QuantumCircuit(2)
        qc.append(gate, [0, 1])

        psi0 = Statevector.from_label("00")
        evolved = psi0.evolve(qc)
        mat = np.asarray(Operator(gate).data)
        expected = Statevector(mat @ psi0.data)

        self.assertTrue(np.allclose(evolved.data, expected.data))

    def test_unitary_parametrized(self):
        cases = [
            ("X", 1),
            ("Y", 1),
            ("Z", 1),
            ("XX", 2),
            ("YY", 2),
            ("ZZ", 2),
            ("XY", 2),
            ("YZ", 2),
            ("XXX", 3),
            ("ZZZ", 3),
        ]

        theta = 1.23
        for pauli, n_qubits in cases:
            with self.subTest(pauli=pauli, n_qubits=n_qubits):
                gate = PauliProductRotationGate(Pauli(pauli), theta)
                mat = np.asarray(Operator(gate).data)
                self.assertEqual(mat.shape, (2**n_qubits, 2**n_qubits))
                self.assertTrue(np.allclose(mat.conj().T @ mat, np.eye(2**n_qubits)))


if __name__ == "__main__":
    unittest.main()
