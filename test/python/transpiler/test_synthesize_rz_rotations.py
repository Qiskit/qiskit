# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the SynthesizeRZRotations pass"""

import numpy as np

from ddt import ddt, data

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RZGate
from qiskit.quantum_info import Operator
from qiskit.quantum_info import get_clifford_gate_names
from qiskit.circuit import Parameter
from qiskit.synthesis import gridsynth_rz
from qiskit.transpiler.passes.synthesis import SynthesizeRZRotations

from test import QiskitTestCase, combine  # pylint: disable=wrong-import-order

# pylint: disable=expression-not-assigned

# Set of single-qubit Clifford gates
CLIFFORD_GATES_1Q_SET = {"id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg"}

# Set of Clifford+T gates
CLIFFORD_T_GATES_SET = set(get_clifford_gate_names() + ["t", "tdg"])


@ddt
class TestSynthesizeRzRotations(QiskitTestCase):
    """Test Synthesize Rz rotations method"""

    def test_synthesize_rz_rotations(self):
        """Test that synthesize_rz_rotations works correctly."""
        num_trials = 40
        for angle in np.linspace(-2 * np.pi, 2 * np.pi, num_trials):
            with self.subTest(angle=angle):
                # Approximate RZ-rotation
                qc = QuantumCircuit(1)
                qc.rz(angle, 0)
                synthesized_circ = SynthesizeRZRotations()(qc)
                # Check the operators are (almost) equal
                self.assertEqual(Operator(synthesized_circ), Operator(RZGate(angle)))

    @data(10, -10)
    def test_synthesize_rz_with_nonstandard_angles(self, angle):
        """Test that synthesize_rz_rotations works correctly."""
        # Approximate RZ-rotation
        qc = QuantumCircuit(1)
        qc.rz(angle, 0)
        synthesized_circ = SynthesizeRZRotations()(qc)
        # Check the operators are (almost) equal
        self.assertEqual(Operator(synthesized_circ), Operator(RZGate(angle)))

    @data(1e-9, 1e-10, 1e-11)
    def test_synthesize_rz_with_approximation_degree(self, epsilon):
        """Test that synthesize_rz_rotations works correctly."""
        approximation_degree = 1 - epsilon
        # Approximate RZ-rotation
        qc = QuantumCircuit(1)
        angle = np.random.uniform(0, 4 * np.pi)
        qc.rz(angle, 0)
        synthesized_circ = SynthesizeRZRotations(approximation_degree=approximation_degree)(qc)
        # Check the operators are (almost) equal
        self.assertEqual(Operator(synthesized_circ), Operator(RZGate(angle)))

    @data(
        0.99,
        0.999,
        0.9999,
        0.99999,
        0.999999,
        0.9999999,
        0.99999999,
        0.999999999,
        0.9999999999,
        0.99999999999,
        0.999999999999,
    )
    def test_approximation_error(self, approximation_degree):
        """Test that the argument ``approximation_degree`` works correctly,"""
        qc = QuantumCircuit(1)
        theta = np.random.uniform(0, 4 * np.pi)
        qc.rz(theta, 0)
        approximate_circuit = SynthesizeRZRotations(approximation_degree)(qc)
        error_matrix = Operator(RZGate(theta)).data - Operator(approximate_circuit).data
        spectral_norm = np.linalg.norm(error_matrix, 2)
        self.assertLessEqual(spectral_norm, 1 - approximation_degree)

    def test_t_counts(self):
        """Test if the expected t-counts are accurate."""
        qc = QuantumCircuit(1)
        qc.rz(1.0, 0)
        approximation_degrees = [0.999999, 0.99999999, 0.9999999999]
        # t_expected = [62, 81, 105]
        t_expected_circs = [gridsynth_rz(1.0, (1 - aps) / 2) for aps in approximation_degrees]
        t_expected = [
            t_expected_circs[i].count_ops().get("t", 0) for i in range(len(approximation_degrees))
        ]
        for ads, t_expect in zip(approximation_degrees, t_expected):
            with self.subTest(eps=ads, t_expect=t_expect):
                qct = SynthesizeRZRotations(ads)(qc)
                t_count = qct.count_ops().get("t", 0)
                self.assertLessEqual(t_count, t_expect)

    def test_param_angle(self):
        """ "Test to see if parametrized angles remain unaffected"""
        p = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rz(p, 0)
        synthesized_circ = SynthesizeRZRotations()(qc)
        self.assertEqual(qc, synthesized_circ)

    def test_synth_rz_deterministic(self):
        """Test that calling synthesize_rz_rotations multiple times produces the same circuit."""
        angle = 1.2345
        num_trials = 10

        qc = QuantumCircuit(1)
        qc.rz(angle, 0)
        approximate_circuits = [SynthesizeRZRotations()(qc) for _ in range(num_trials)]

        for idx in range(1, len(approximate_circuits)):
            self.assertEqual(approximate_circuits[idx], approximate_circuits[0])

    @data(8, 10)
    def test_angle_canonicalization(self, num_qubits):
        """Test that the angle canonicalization and corresponding
        phase, gate updates in synthesize_rz_rotations works correctly."""
        qc = QuantumCircuit(num_qubits)
        angle = np.random.uniform(0, np.pi / 2)
        [qc.rz(_ * np.pi / 2 + angle, _) for _ in range(num_qubits)]
        synthesized_circ = SynthesizeRZRotations()(qc)
        # Check the operators are (almost) equal
        self.assertEqual(Operator(synthesized_circ), Operator(qc))

    @combine(num_qubits=[5, 8], depth=[6, 10])
    def test_dag_traversal_logic(self, num_qubits, depth):
        """Test that synthesize_rz_rotations works correctly for larger circuits."""
        qc = QuantumCircuit(num_qubits)
        [
            [qc.rz(np.random.uniform(0, 4 * np.pi), _) for _ in range(num_qubits)]
            for _ in range(depth)
        ]
        synthesized_circ = SynthesizeRZRotations()(qc)
        # Check the operators are (almost) equal
        self.assertEqual(Operator(synthesized_circ), Operator(qc))
        
