# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Test StateVectorSimulatorPy."""

import unittest

import numpy as np

from qiskit.providers.basicaer import StatevectorSimulatorPy
from qiskit.test import ReferenceCircuits
from qiskit.test import providers
from qiskit import QuantumRegister, QuantumCircuit, execute
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info import state_fidelity


class StatevectorSimulatorTest(providers.BackendTestCase):
    """Test BasicAer statevector simulator."""

    backend_cls = StatevectorSimulatorPy
    circuit = None

    def test_run_circuit(self):
        """Test final state vector for single circuit run."""
        # Set test circuit
        self.circuit = ReferenceCircuits.bell_no_measure()
        # Execute
        result = super().test_run_circuit()
        actual = result.get_statevector(self.circuit)

        # state is 1/sqrt(2)|00> + 1/sqrt(2)|11>, up to a global phase
        self.assertAlmostEqual((abs(actual[0])) ** 2, 1 / 2)
        self.assertEqual(actual[1], 0)
        self.assertEqual(actual[2], 0)
        self.assertAlmostEqual((abs(actual[3])) ** 2, 1 / 2)

    def test_measure_collapse(self):
        """Test final measurement collapses statevector"""
        # Set test circuit
        self.circuit = ReferenceCircuits.bell()
        # Execute
        result = super().test_run_circuit()
        actual = result.get_statevector(self.circuit)

        # The final state should be EITHER |00> OR |11>
        diff_00 = np.linalg.norm(np.array([1, 0, 0, 0]) - actual) ** 2
        diff_11 = np.linalg.norm(np.array([0, 0, 0, 1]) - actual) ** 2
        success = np.allclose([diff_00, diff_11], [0, 2]) or np.allclose([diff_00, diff_11], [2, 0])
        # state is 1/sqrt(2)|00> + 1/sqrt(2)|11>, up to a global phase
        self.assertTrue(success)

    def test_unitary(self):
        """Test unitary gate instruction"""
        num_trials = 10
        max_qubits = 3
        # Test 1 to max_qubits for random n-qubit unitary gate
        for i in range(max_qubits):
            num_qubits = i + 1
            psi_init = np.zeros(2 ** num_qubits)
            psi_init[0] = 1.0
            qr = QuantumRegister(num_qubits, "qr")
            for _ in range(num_trials):
                # Create random unitary
                unitary = random_unitary(2 ** num_qubits)
                # Compute expected output state
                psi_target = unitary.data.dot(psi_init)
                # Simulate output on circuit
                circuit = QuantumCircuit(qr)
                circuit.unitary(unitary, qr)
                job = execute(circuit, self.backend)
                result = job.result()
                psi_out = result.get_statevector(0)
                fidelity = state_fidelity(psi_target, psi_out)
                self.assertGreater(fidelity, 0.999)

    def test_global_phase(self):
        """Test global_phase"""
        n_qubits = 4
        qr = QuantumRegister(n_qubits)
        circ = QuantumCircuit(qr)
        circ.x(qr)
        circ.global_phase = 0.5
        self.circuit = circ
        result = super().test_run_circuit()
        actual = result.get_statevector(self.circuit)
        expected = np.exp(1j * circ.global_phase) * np.repeat([[0], [1]], [n_qubits ** 2 - 1, 1])
        self.assertTrue(np.allclose(actual, expected))

    def test_global_phase_composite(self):
        """Test global_phase"""
        n_qubits = 4
        qr = QuantumRegister(n_qubits)
        circ = QuantumCircuit(qr)
        circ.x(qr)
        circ.global_phase = 0.5
        gate = circ.to_gate()

        comp = QuantumCircuit(qr)
        comp.append(gate, qr)
        comp.global_phase = 0.1
        self.circuit = comp
        result = super().test_run_circuit()
        actual = result.get_statevector(self.circuit)
        expected = np.exp(1j * 0.6) * np.repeat([[0], [1]], [n_qubits ** 2 - 1, 1])
        self.assertTrue(np.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
