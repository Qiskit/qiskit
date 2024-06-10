# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test random circuit generation utility."""
from collections import defaultdict
import rustworkx as rx
import numpy as np
from qiskit.circuit import QuantumCircuit, ClassicalRegister, Clbit
from qiskit.circuit import Measure
from qiskit.circuit.random import random_circuit
from qiskit.circuit.random.utils import random_circuit_from_graph
from qiskit.converters import circuit_to_dag
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestCircuitRandom(QiskitTestCase):
    """Testing qiskit.circuit.random"""

    def test_simple_random(self):
        """Test creating a simple random circuit."""
        circ = random_circuit(num_qubits=5, depth=4)
        self.assertIsInstance(circ, QuantumCircuit)
        self.assertEqual(circ.width(), 5)
        self.assertEqual(circ.depth(), 4)

    def test_random_depth_0(self):
        """Test random depth 0 circuit."""
        circ = random_circuit(num_qubits=1, depth=0)
        self.assertEqual(circ.width(), 1)
        self.assertEqual(circ.depth(), 0)

    def test_random_measure(self):
        """Test random circuit with final measurement."""
        num_qubits = depth = 3
        circ = random_circuit(num_qubits, depth, measure=True)
        self.assertEqual(circ.width(), 2 * num_qubits)
        dag = circuit_to_dag(circ)
        for nd in list(dag.topological_op_nodes())[-num_qubits:]:
            self.assertIsInstance(nd.op, Measure)

    def test_random_circuit_conditional_reset(self):
        """Test generating random circuits with conditional and reset."""
        num_qubits = 1
        depth = 100
        circ = random_circuit(num_qubits, depth, conditional=True, reset=True, seed=5)
        self.assertEqual(circ.width(), 2 * num_qubits)
        self.assertIn("reset", circ.count_ops())

    def test_large_conditional(self):
        """Test that conditions do not fail with large conditionals.  Regression test of gh-6994."""
        # The main test is that this call actually returns without raising an exception.
        circ = random_circuit(64, 2, conditional=True, seed=0)
        # Test that at least one instruction had a condition generated.  It's possible that this
        # fails due to very bad luck with the random seed - if so, change the seed to ensure that a
        # condition _is_ generated, because we need to test that generation doesn't error.
        conditions = (getattr(instruction.operation, "condition", None) for instruction in circ)
        conditions = [x for x in conditions if x is not None]
        self.assertNotEqual(conditions, [])
        for register, value in conditions:
            self.assertIsInstance(register, (ClassicalRegister, Clbit))
            # Condition values always have to be Python bigints (of which `bool` is a subclass), not
            # any of Numpy's fixed-width types, for example.
            self.assertIsInstance(value, int)

    def test_random_mid_circuit_measure_conditional(self):
        """Test random circuit with mid-circuit measurements for conditionals."""
        num_qubits = depth = 2
        circ = random_circuit(num_qubits, depth, conditional=True, seed=16)
        self.assertEqual(circ.width(), 2 * num_qubits)
        op_names = [instruction.operation.name for instruction in circ]
        # Before a condition, there needs to be measurement in all the qubits.
        self.assertEqual(4, len(op_names))
        self.assertEqual(["measure"] * num_qubits, op_names[1 : 1 + num_qubits])
        conditions = [
            bool(getattr(instruction.operation, "condition", None)) for instruction in circ
        ]
        self.assertEqual([False, False, False, True], conditions)

    def test_random_circuit_num_operand_distribution(self):
        """Test that num_operand_distribution argument generates gates in correct proportion"""
        num_qubits = 50
        depth = 300
        num_op_dist = {2: 0.25, 3: 0.25, 1: 0.25, 4: 0.25}
        circ = random_circuit(
            num_qubits, depth, num_operand_distribution=num_op_dist, seed=123456789
        )
        total_gates = circ.size()
        self.assertEqual(circ.width(), num_qubits)
        self.assertEqual(circ.depth(), depth)
        gate_qubits = [instruction.operation.num_qubits for instruction in circ]
        gate_type_counter = np.bincount(gate_qubits, minlength=5)
        for gate_type, prob in sorted(num_op_dist.items()):
            self.assertAlmostEqual(prob, gate_type_counter[gate_type] / total_gates, delta=0.1)

    def test_random_circuit_2and3_qubit_gates_only(self):
        """
        Test that the generated random circuit only has 2 and 3 qubit gates,
        while disallowing 1-qubit and 4-qubit gates if
        num_operand_distribution = {2: some_prob, 3: some_prob}
        """
        num_qubits = 10
        depth = 200
        num_op_dist = {2: 0.5, 3: 0.5}
        circ = random_circuit(num_qubits, depth, num_operand_distribution=num_op_dist, seed=200)
        total_gates = circ.size()
        gate_qubits = [instruction.operation.num_qubits for instruction in circ]
        gate_type_counter = np.bincount(gate_qubits, minlength=5)
        # Testing that the distribution of 2 and 3 qubit gate matches with given distribution
        for gate_type, prob in sorted(num_op_dist.items()):
            self.assertAlmostEqual(prob, gate_type_counter[gate_type] / total_gates, delta=0.1)
        # Testing that there are no 1-qubit gate and 4-qubit in the generated random circuit
        self.assertEqual(gate_type_counter[1], 0.0)
        self.assertEqual(gate_type_counter[4], 0.0)

    def test_random_circuit_3and4_qubit_gates_only(self):
        """
        Test that the generated random circuit only has 3 and 4 qubit gates,
        while disallowing 1-qubit and 2-qubit gates if
        num_operand_distribution = {3: some_prob, 4: some_prob}
        """
        num_qubits = 10
        depth = 200
        num_op_dist = {3: 0.5, 4: 0.5}
        circ = random_circuit(
            num_qubits, depth, num_operand_distribution=num_op_dist, seed=11111111
        )
        total_gates = circ.size()
        gate_qubits = [instruction.operation.num_qubits for instruction in circ]
        gate_type_counter = np.bincount(gate_qubits, minlength=5)
        # Testing that the distribution of 3 and 4 qubit gate matches with given distribution
        for gate_type, prob in sorted(num_op_dist.items()):
            self.assertAlmostEqual(prob, gate_type_counter[gate_type] / total_gates, delta=0.1)
        # Testing that there are no 1-qubit gate and 2-qubit in the generated random circuit
        self.assertEqual(gate_type_counter[1], 0.0)
        self.assertEqual(gate_type_counter[2], 0.0)

    def test_random_circuit_with_zero_distribution(self):
        """
        Test that the generated random circuit only has 3 and 4 qubit gates,
        while disallowing 1-qubit and 2-qubit gates if
        num_operand_distribution = {1: 0.0, 2: 0.0, 3: some_prob, 4: some_prob}
        """
        num_qubits = 10
        depth = 200
        num_op_dist = {1: 0.0, 2: 0.0, 3: 0.5, 4: 0.5}
        circ = random_circuit(num_qubits, depth, num_operand_distribution=num_op_dist, seed=12)
        total_gates = circ.size()
        gate_qubits = [instruction.operation.num_qubits for instruction in circ]
        gate_type_counter = np.bincount(gate_qubits, minlength=5)
        # Testing that the distribution of 3 and 4 qubit gate matches with given distribution
        for gate_type, prob in sorted(num_op_dist.items()):
            self.assertAlmostEqual(prob, gate_type_counter[gate_type] / total_gates, delta=0.1)
        # Testing that there are no 1-qubit gate and 2-qubit in the generated random circuit
        self.assertEqual(gate_type_counter[1], 0.0)
        self.assertEqual(gate_type_counter[2], 0.0)


class TestRandomCircuitFromGraph(QiskitTestCase):
    """Testing random_circuit_from_graph from
    qiskit.circuit.random.utils.py"""

    def setUp(self):
        super().setUp()
        n_q = 10
        pydi_graph = rx.PyDiGraph()
        pydi_graph.add_nodes_from(range(n_q))
        # Some arbitrary coupling map
        cp_map = [
            (0, 2, None),
            (1, 3, None),
            (2, 4, None),
            (3, 4, None),
            (5, 7, None),
            (4, 7, None),
            (7, 9, None),
            (5, 8, None),
            (6, 9, None),
        ]
        pydi_graph.add_edges_from(cp_map)
        self.interaction_graph = (pydi_graph, None, None, None)

    def test_simple_random(self):
        """Test creating a simple random circuit."""
        circ = random_circuit_from_graph(
            interaction_graph=self.interaction_graph, min_2q_gate_per_edge=1
        )
        self.assertIsInstance(circ, QuantumCircuit)
        self.assertEqual(circ.width(), 10)

    def test_min_times_qubit_pair_usage(self):
        """the depth parameter specifies how often each qubit-pair must at
        least be used in a two-qubit gate before the circuit is returned"""

        freq = 1
        qc = random_circuit_from_graph(
            interaction_graph=self.interaction_graph, min_2q_gate_per_edge=freq, seed=0
        )
        dag = circuit_to_dag(qc)
        count_register = defaultdict(int)

        for wire in dag.wires:
            for node in dag.nodes_on_wire(wire, only_ops=True):
                if node.op.name == "measure" or node.op.num_qubits < 2:
                    continue
                key = node.sort_key
                control, target = key.split(",")
                count_register[(control, target)] += 1

        for occurence in count_register.values():
            self.assertLessEqual(freq, occurence)

    def test_random_measure(self):
        """Test random circuit with final measurement."""
        qc = random_circuit_from_graph(
            interaction_graph=self.interaction_graph, min_2q_gate_per_edge=1, measure=True
        )
        self.assertIn("measure", qc.count_ops())

    def test_random_circuit_conditional_reset(self):
        """Test generating random circuits with conditional and reset."""
        # Presence of 'reset' in the circuit is probabilistic, at seed 0 reset exists in circuit.
        qc = random_circuit_from_graph(
            interaction_graph=self.interaction_graph,
            min_2q_gate_per_edge=1,
            conditional=True,
            reset=True,
            seed=486,  # Do not change the seed or the args.
            insert_1q_oper=True,
        )
        self.assertIn("reset", qc.count_ops())

    def test_large_conditional_weighted_qubits(self):
        """Test that conditions do not fail with large conditionals.  Regression test of gh-6994."""
        # This is to test the call actually returns without raising an exception.
        # In this case every qubit-pair is associated with a probability of being selected.
        n_q = 10
        pydi_graph = rx.PyDiGraph()
        pydi_graph.add_nodes_from(range(n_q))
        # Some arbitrary coupling map
        # ( control, target, probability-of-being-selected )
        cp_map = [
            (0, 2, 0.1),
            (1, 3, 0.15),
            (2, 4, 0.15),
            (3, 4, 0.1),
            (5, 7, 0.13),
            (4, 7, 0.07),
            (7, 9, 0.1),
            (5, 8, 0.1),
            (6, 9, 0.1),
        ]
        pydi_graph.add_edges_from(cp_map)
        interaction_graph = (pydi_graph, None, None, None)

        circ = random_circuit_from_graph(
            interaction_graph=interaction_graph,
            min_2q_gate_per_edge=1,
            measure=True,
            conditional=True,
            reset=True,
            seed=778,  # Do not change the seed or the args.
        )
        # Test that at least one instruction having a conditional is generated.  Keep seed as 0.
        # Do not change the function signature.
        conditions = (getattr(instruction.operation, "condition", None) for instruction in circ)
        conditions = [x for x in conditions if x is not None]
        self.assertNotEqual(conditions, [])
        for register, value in conditions:
            self.assertIsInstance(register, (ClassicalRegister, Clbit))
            # Condition values always have to be Python bigints (of which `bool` is a subclass), not
            # any of Numpy's fixed-width types, for example.
            self.assertIsInstance(value, int)

    def test_large_conditional(self):
        """Test that conditions do not fail with large conditionals.  Regression test of gh-6994."""
        # This is to test the call actually returns without raising an exception.
        circ = random_circuit_from_graph(
            interaction_graph=self.interaction_graph,
            min_2q_gate_per_edge=1,
            measure=True,
            conditional=True,
            reset=True,
            seed=0,  # Do not change the seed or the args.
        )
        # Test that at least one instruction having a conditional is generated.  Keep seed as 0.
        # Do not change the function signature.
        conditions = (getattr(instruction.operation, "condition", None) for instruction in circ)
        conditions = [x for x in conditions if x is not None]
        self.assertNotEqual(conditions, [])
        for register, value in conditions:
            self.assertIsInstance(register, (ClassicalRegister, Clbit))
            # Condition values always have to be Python bigints (of which `bool` is a subclass), not
            # any of Numpy's fixed-width types, for example.
            self.assertIsInstance(value, int)

    def test_random_mid_circuit_measure_conditional(self):
        """Test random circuit with mid-circuit measurements for conditionals."""
        qc = random_circuit_from_graph(
            interaction_graph=self.interaction_graph,
            min_2q_gate_per_edge=1,
            measure=True,
            conditional=True,
            reset=True,
            seed=0,
        )
        dag = circuit_to_dag(qc)

        # Before a condition, there needs to be measurement of atleast one of the qubits.
        measure_at = None
        condition_at = None
        for qubit_idx, wire in enumerate(dag.wires):
            if condition_at is not None and measure_at is not None:
                break
            for layer_num, dag_op_node in enumerate(dag.nodes_on_wire(wire, only_ops=True)):
                if condition_at is None and getattr(dag_op_node.op, "condition", None) is not None:
                    condition_at = {"qubit_idx": qubit_idx + 1, "layer_no": layer_num + 1}
                elif measure_at is None and dag_op_node.op.name == "measure":
                    measure_at = {"qubit_idx": qubit_idx + 1, "layer_no": layer_num + 1}

        self.assertGreater(condition_at["layer_no"], measure_at["layer_no"])
