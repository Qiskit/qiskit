# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the evolution gate."""

import scipy

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import EvolutionGate
from qiskit.circuit.library.evolution.lie_trotter import LieTrotter
from qiskit.circuit.library.evolution.suzuki_trotter import SuzukiTrotter
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.opflow import I, X, Y, Z
from qiskit.quantum_info import Operator, SparsePauliOp


class TestEvolutionGate(QiskitTestCase):
    """Test the evolution gate."""

    def test_default_decomposition(self):
        """Test the default decomposition."""
        op = (X ^ 3) + (Y ^ 3) + (Z ^ 3)
        time = 0.123

        matrix = op.to_matrix()
        evolved = scipy.linalg.expm(-1j * time * matrix)

        evo_gate = EvolutionGate(op, time)

        self.assertTrue(Operator(evo_gate).equiv(evolved))

    def test_lie_trotter(self):
        """Test constructing the circuit with Lie Trotter decomposition."""
        op = (X ^ 3) + (Y ^ 3) + (Z ^ 3)
        time = 0.123
        reps = 4
        evo_gate = EvolutionGate(op, time, synthesis=LieTrotter(reps=reps))
        decomposed = evo_gate.definition.decompose()
        self.assertEqual(decomposed.count_ops()["cx"], reps * 3 * 4)

    def test_suzuki_trotter(self):
        """Test constructing the circuit with Lie Trotter decomposition."""
        op = (X ^ 3) + (Y ^ 3) + (Z ^ 3)
        time = 0.123
        reps = 4
        for order in [2, 4, 5]:
            if order == 2:
                expected_cx = reps * 5 * 4
            elif order % 2 == 0:
                # recurse (order - 2) / 2 times, base case has 5 blocks with 4 CX each
                expected_cx = reps * 5 ** ((order - 2) / 2) * 5 * 4
            else:
                # recurse (order - 1) / 2 times, base case has 3 blocks with 4 CX each
                expected_cx = reps * 5 ** ((order - 1) / 2) * 3 * 4

            with self.subTest(order=order):
                evo_gate = EvolutionGate(op, time, synthesis=SuzukiTrotter(order=order, reps=reps))
                decomposed = evo_gate.definition.decompose()
                self.assertEqual(decomposed.count_ops()["cx"], expected_cx)

    def test_suzuki_trotter_manual(self):
        """Test the evolution circuit of Suzuki Trotter against a manually constructed circuit."""
        op = X + Y
        time = 0.1
        reps = 1
        evo_gate = EvolutionGate(op, time, synthesis=SuzukiTrotter(order=4, reps=reps))

        # manually construct expected evolution
        expected = QuantumCircuit(1)
        p_4 = 1 / (4 - 4 ** (1 / 3))  # coefficient for reduced time from Suzuki paper
        for _ in range(2):
            # factor of 1/2 already included with factor 1/2 in rotation gate
            expected.rx(p_4 * time, 0)
            expected.ry(2 * p_4 * time, 0)
            expected.rx(p_4 * time, 0)

        expected.rx((1 - 4 * p_4) * time, 0)
        expected.ry(2 * (1 - 4 * p_4) * time, 0)
        expected.rx((1 - 4 * p_4) * time, 0)

        for _ in range(2):
            expected.rx(p_4 * time, 0)
            expected.ry(2 * p_4 * time, 0)
            expected.rx(p_4 * time, 0)

        decomposed = evo_gate.definition.decompose()
        self.assertEqual(decomposed, expected)

    def test_passing_grouped_paulis(self):
        """Test passing a list of already grouped Paulis."""
        grouped_ops = [(X ^ Y) + (Y ^ X), (Z ^ I) + (Z ^ Z) + (I ^ Z), (X ^ X)]
        evo_gate = EvolutionGate(grouped_ops, time=0.12, synthesis=LieTrotter())
        decomposed = evo_gate.definition.decompose()
        self.assertEqual(decomposed.count_ops()["rz"], 6)

    def test_list_from_grouped_paulis(self):
        """Test getting a string representation from grouped Paulis."""
        grouped_ops = [(X ^ Y) + (Y ^ X), (Z ^ I) + (Z ^ Z) + (I ^ Z), (X ^ X)]
        evo_gate = EvolutionGate(grouped_ops, time=0.12, synthesis=LieTrotter())

        pauli_strings = []
        for op in evo_gate.operator:
            if isinstance(op, SparsePauliOp):
                pauli_strings.append(op.to_list())
            else:
                pauli_strings.append([(str(op), 1 + 0j)])

        expected = [
            [("XY", 1 + 0j), ("YX", 1 + 0j)],
            [("ZI", 1 + 0j), ("ZZ", 1 + 0j), ("IZ", 1 + 0j)],
            [("XX", 1 + 0j)],
        ]
        self.assertListEqual(pauli_strings, expected)

    def test_dag_conversion(self):
        """Test constructing a circuit with evolutions yields a DAG with evolution blocks."""
        time = Parameter("t")
        evo = EvolutionGate(X ^ 2, time=time, synthesis=LieTrotter())

        circuit = QuantumCircuit(2)
        circuit.h(circuit.qubits)
        circuit.append(evo, circuit.qubits)
        circuit.cx(0, 1)

        dag = circuit_to_dag(circuit)

        expected_ops = {"h", "cx", "EvolutionGate"}
        ops = set(node.op.name for node in dag.op_nodes())

        self.assertEqual(ops, expected_ops)
