# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
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
                qc = QuantumCircuit(1)
                qc.rz(angle, 0)
                synthesized_circ = SynthesizeRZRotations()(qc)
                self.assertEqual(Operator(synthesized_circ), Operator(RZGate(angle)))

    @data(10, -10)
    def test_synthesize_rz_with_nonstandard_angles(self, angle):
        """Test that synthesize_rz_rotations works correctly."""
        # Approximate RZ-rotation
        qc = QuantumCircuit(1)
        qc.rz(angle, 0)
        synthesized_circ = SynthesizeRZRotations()(qc)
        self.assertEqual(Operator(synthesized_circ), Operator(RZGate(angle)))

    @data(
        1 - 1e-2,
        1 - 1e-3,
        1 - 1e-4,
        1 - 1e-5,
        1 - 1e-6,
        1 - 1e-7,
        1 - 1e-8,
        1 - 1e-9,
        1 - 1e-10,
        1 - 1e-11,
        1 - 1e-12,
    )
    def test_approximation_error(self, approximation_degree):
        """Test that the argument ``approximation_degree`` works correctly,"""
        qc = QuantumCircuit(1)
        theta = 2.3579
        qc.rz(theta, 0)
        approximate_circuit = SynthesizeRZRotations(approximation_degree)(qc)
        error_matrix = Operator(RZGate(theta)).data - Operator(approximate_circuit).data
        spectral_norm = np.linalg.norm(error_matrix, 2)
        self.assertLessEqual(spectral_norm, 1 - approximation_degree)

    @data(
        1 - 1e-2,
        1 - 1e-3,
        1 - 1e-4,
        1 - 1e-5,
        1 - 1e-6,
        1 - 1e-7,
        1 - 1e-8,
        1 - 1e-9,
        1 - 1e-10,
        1 - 1e-11,
        1 - 1e-12,
    )
    def test_t_counts_given_approximation_degree(self, approximation_degree):
        """Test if the expected t-counts provided by the pass are consistent with
        the underlying synthesis method.
        """
        qc = QuantumCircuit(1)
        qc.rz(1.0, 0)

        qct = SynthesizeRZRotations(approximation_degree=approximation_degree)(qc)
        t_count = qct.count_ops().get("t", 0)

        # Dividing by 2 because this is how SynthesizeRZRotations splits total error budget
        expected_circ = gridsynth_rz(1.0, (1 - approximation_degree) / 2)
        t_expect = expected_circ.count_ops().get("t", 0)

        self.assertEqual(t_count, t_expect)

    def test_cache_error(self):
        """Test that the cache_error argument works as expected
        when both synthesis_error and cache_error are given.
        """
        qc = QuantumCircuit(1)
        qc.rz(0.0, 0)
        qc.rz(0.1, 0)
        qc.rz(0.2, 0)
        qc.rz(0.3, 0)

        # Sets a very high cache_error. Because both synthesis_error and
        # cache_error are given, we expect them to be used instead of the
        # approximation degree.
        # Exploiting our knowledge of the inner workings
        # of the SynthesizeRZRotations pass, the RZ(0.0, 0) gate will get synthesized
        # first (requiring no T-gates), while the other gates should reuse its synthesis
        # result (hence also requiring no T-gates).
        qct = SynthesizeRZRotations(synthesis_error=1e-10, cache_error=0.5)(qc)
        t_count = qct.count_ops().get("t", 0)
        self.assertEqual(t_count, 0)

    def test_synthesis_error(self):
        """Test that the synthesis_error argument works as expected
        when both synthesis_error and cache_error are given.
        """
        angle = 1.2345

        qc = QuantumCircuit(1)
        qc.rz(angle, 0)

        # Sets a low synthesis error. Because both synthesis_error and
        # cache_error are given, we expect them to be used instead of the
        # approximation degree.
        qct = SynthesizeRZRotations(
            approximation_degree=1e-10, synthesis_error=1e-2, cache_error=0.0
        )(qc)
        t_count = qct.count_ops().get("t", 0)

        expected_circ = gridsynth_rz(angle, 1e-2)
        t_expect = expected_circ.count_ops().get("t", 0)

        self.assertEqual(t_count, t_expect)

    def test_param_angle(self):
        """Test to see if parametrized angles remain unaffected"""
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
        qcs = [QuantumCircuit(1) for _ in range(num_qubits)]
        angle = 1.23579
        _ = [qcs[_].rz(_ * np.pi / 2 + angle, 0) for _ in range(num_qubits)]
        synthesized_circs = [SynthesizeRZRotations()(qcs[_]) for _ in range(num_qubits)]
        qc_big = qcs[0].copy()
        for _ in range(1, num_qubits):
            qc_big = qc_big.tensor(qcs[_])
        qc_big_synth = SynthesizeRZRotations()(qc_big)
        [
            self.assertEqual(Operator(synthesized_circs[_]), Operator(qcs[_]))
            for _ in range(num_qubits)
        ]
        self.assertEqual(Operator(qc_big_synth), Operator(qc_big))

    @combine(num_qubits=[5, 8], depth=[6, 10])
    def test_dag_traversal_logic(self, num_qubits, depth):
        """Test that synthesize_rz_rotations works correctly for larger circuits."""
        qc = QuantumCircuit(num_qubits)
        _ = [[qc.rz(3.57921, _) for _ in range(num_qubits)] for _ in range(depth)]
        synthesized_circ = SynthesizeRZRotations()(qc)
        self.assertEqual(Operator(synthesized_circ), Operator(qc))
