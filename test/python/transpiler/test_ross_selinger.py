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

"""Test the Ross-Selinger synthesis and plugin."""

import unittest
import numpy as np

from ddt import ddt, data

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RZGate
from qiskit.quantum_info import Operator
from qiskit.quantum_info import get_clifford_gate_names
from qiskit.quantum_info.random import random_unitary
from qiskit.synthesis import approximate_rz_rotation, approximate_1q_unitary
from qiskit.converters import dag_to_circuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import UnitarySynthesis, Collect1qRuns, ConsolidateBlocks
from qiskit.transpiler.passes.synthesis import RossSelingerSynthesis

from test import QiskitTestCase  # pylint: disable=wrong-import-order


# Set of single-qubit Clifford gates
CLIFFORD_GATES_1Q_SET = {"id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg"}

# Set of Clifford+T gates
CLIFFORD_T_GATES_SET = set(get_clifford_gate_names() + ["t", "tdg"])


@ddt
class TestRossSelingerSynthesis(QiskitTestCase):
    """Test Ross-Selinger synthesis methods."""

    def test_approximate_rz_rotation_correct(self):
        """Test that approximate_rz_rotation works correctly."""
        num_trials = 40
        for angle in np.linspace(-2 * np.pi, 2 * np.pi, num_trials):
            with self.subTest(angle=angle):
                # Approximate RZ-rotation
                approximate_circuit = approximate_rz_rotation(angle, 1e-10)
                # Check the operators are (almost) equal
                self.assertEqual(Operator(approximate_circuit), Operator(RZGate(angle)))

    @data(10, -10)
    def test_approximate_rz_rotation_with_nonstandard_angles(self, angle):
        """Test that approximate_rz_rotation works correctly."""
        # Approximate RZ-rotation
        approximate_circuit = approximate_rz_rotation(angle, 1e-10)
        # Check the operators are (almost) equal
        self.assertEqual(Operator(approximate_circuit), Operator(RZGate(angle)))

    def test_approximate_1q_unitary_correct(self):
        """Test that approximate_1q_unitary works correctly."""
        num_trials = 50
        for seed in range(num_trials):
            with self.subTest(seed=seed):
                # Create a random 1q unitary.
                unitary = random_unitary(2, seed=seed)
                # Approximate unitary
                approximate_circuit = approximate_1q_unitary(unitary.data, 1e-10)
                # Check the operators are (almost) equal
                self.assertEqual(Operator(approximate_circuit), Operator(unitary))

    def test_approximate_rz_rotation_deterministic(self):
        """Test that calling approximate_rz_rotation multiple times produces the same circuit."""
        angle = 1.2345
        num_trials = 10
        approximate_circuits = [approximate_rz_rotation(angle, 1e-10) for _ in range(num_trials)]

        for idx in range(1, len(approximate_circuits)):
            self.assertEqual(approximate_circuits[idx], approximate_circuits[0])

    def test_approximate_1q_unitary_deterministic(self):
        """Test that calling approximate_1q_unitary multiple times produces the same circuit."""
        unitary = random_unitary(2, seed=12345)
        num_trials = 10
        approximate_circuits = [
            approximate_1q_unitary(unitary.data, 1e-10) for _ in range(num_trials)
        ]

        for idx in range(1, len(approximate_circuits)):
            self.assertEqual(approximate_circuits[idx], approximate_circuits[0])

    def test_identity_matrix(self):
        """Test that the Ross-Selinger algorithm does not return T-gates when approximating
        the identity matrix.
        """
        # Note that the algorithm may produce CLifford gates.
        circuit = QuantumCircuit(1)
        matrix = Operator(circuit).data
        approximate_circuit = approximate_1q_unitary(matrix)
        self.assertLessEqual(set(approximate_circuit.count_ops()), CLIFFORD_GATES_1Q_SET)

    # ToDo: finish this test when rsgridsynth is able to hangle T-gate.
    # def test_t_matrix(self):
    #     """Test what happens on a circuit with a single T-gate"""
    #     ...

    # ToDo: add more transpile-level tests when Ross-Selinger is included in the default plugin.


@ddt
class TestRossSelingerPlugin(QiskitTestCase):
    """Test the Ross-Selinger unitary synthesis plugin."""

    def test_unitary_synthesis(self):
        """Test the unitary synthesis transpiler pass with Ross-Selinger algorithm."""
        circuit = QuantumCircuit(2)
        circuit.rx(0.8, 0)
        circuit.cx(0, 1)
        circuit.x(1)

        _1q = Collect1qRuns()
        _cons = ConsolidateBlocks()
        _synth = UnitarySynthesis(["h", "t", "tdg"], method="rs")
        passes = PassManager([_1q, _cons, _synth])
        compiled = passes.run(circuit)

        # The approximation should be good enough for the Operator-equality check to pass
        self.assertEqual(Operator(circuit), Operator(compiled))
        self.assertLessEqual(set(compiled.count_ops()), CLIFFORD_T_GATES_SET)

    def test_plugin(self):
        """Test calling the Ross-Selinger plugin directly."""
        circuit = QuantumCircuit(1)
        circuit.rx(0.8, 0)

        unitary = Operator(circuit).data

        plugin = RossSelingerSynthesis()
        compiled_dag = plugin.run(unitary)
        compiled = dag_to_circuit(compiled_dag)

        # The approximation should be good enough for the Operator-equality check to pass
        self.assertEqual(Operator(circuit), Operator(compiled))
        self.assertLessEqual(set(compiled.count_ops()), CLIFFORD_T_GATES_SET)


if __name__ == "__main__":
    unittest.main()
