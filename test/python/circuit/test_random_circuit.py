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
import rustworkx as rx
from qiskit.circuit import QuantumCircuit, ClassicalRegister, Clbit
from qiskit.circuit import Measure
from qiskit.circuit.random import random_circuit
from qiskit.circuit.random.utils import random_circuit_with_graph
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
        circ = random_circuit(num_qubits, depth, conditional=True, seed=4)
        self.assertEqual(circ.width(), 2 * num_qubits)
        op_names = [instruction.operation.name for instruction in circ]
        # Before a condition, there needs to be measurement in all the qubits.
        self.assertEqual(4, len(op_names))
        self.assertEqual(["measure"] * num_qubits, op_names[1 : 1 + num_qubits])
        conditions = [
            bool(getattr(instruction.operation, "condition", None)) for instruction in circ
        ]
        self.assertEqual([False, False, False, True], conditions)


class TestRandomCircuitWithGraph(QiskitTestCase):
    """Testing random_circuit_with_graph from
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
        ]
        pydi_graph.add_edges_from(cp_map)
        self.interaction_graph = (pydi_graph, None, None, None)

    def test_simple_random(self):
        """Test creating a simple random circuit."""
        circ = random_circuit_with_graph(
            interaction_graph=self.interaction_graph, max_num_qubit_usage=2
        )
        self.assertIsInstance(circ, QuantumCircuit)
        self.assertEqual(circ.width(), 10)

    def test_max_times_qubit_usage(self):
        """Test number of gates on a qubit doesn't exceeds the `max_num_qubit_usage`."""
        qc = random_circuit_with_graph(
            interaction_graph=self.interaction_graph, max_num_qubit_usage=4
        )
        # can go for qc.depth() == 4 for all these.
        dag = circuit_to_dag(qc)
        max_usage_per_qubit = [
            len([dag_op_node.name for dag_op_node in dag.nodes_on_wire(wire, only_ops=True)])
            for wire in dag.wires
        ]
        for usage in max_usage_per_qubit:
            self.assertLessEqual(usage, 4)

    def test_random_measure(self):
        """Test random circuit with final measurement."""
        qc = random_circuit_with_graph(
            interaction_graph=self.interaction_graph, max_num_qubit_usage=1, measure=True
        )
        self.assertIn("measure", qc.count_ops())

    def test_random_circuit_conditional_reset(self):
        """Test generating random circuits with conditional and reset."""
        # Presence of 'reset' in the circuit is probabilistic, at seed 0 reset exists in circuit.
        qc = random_circuit_with_graph(
            interaction_graph=self.interaction_graph,
            max_num_qubit_usage=2,
            conditional=True,
            reset=True,
            seed=0,
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
            (1, 3, 0.05),
            (2, 4, 0.15),
            (3, 4, 0.2),
            (5, 7, 0.03),
            (4, 7, 0.07),
            (7, 9, 0.2),
            (5, 8, 0.2),
        ]
        pydi_graph.add_edges_from(cp_map)
        interaction_graph = (pydi_graph, None, None, None)

        circ = random_circuit_with_graph(
            interaction_graph=interaction_graph,
            max_num_qubit_usage=3,
            measure=True,
            conditional=True,
            reset=True,
            seed=2589,
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
        circ = random_circuit_with_graph(
            interaction_graph=self.interaction_graph,
            max_num_qubit_usage=2,
            measure=True,
            conditional=True,
            reset=True,
            seed=0,
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
        qc = random_circuit_with_graph(
            interaction_graph=self.interaction_graph,
            max_num_qubit_usage=2,
            measure=True,
            conditional=True,
            reset=True,
            seed=0,
        )
        dag = circuit_to_dag(qc)

        # Before a condition, there needs to be measurement of atleast one of the qubits.
        measure_at = None
        condition_at = None
        for layer_num, wire in enumerate(dag.wires):
            if condition_at is not None and measure_at is not None:
                break
            for oper_num, dag_op_node in enumerate(dag.nodes_on_wire(wire, only_ops=True)):
                if condition_at is None and getattr(dag_op_node.op, "condition", None) is not None:
                    condition_at = {"layer_no": layer_num + 1, "oper_num": oper_num + 1}
                elif measure_at is None and dag_op_node.op.name == "measure":
                    measure_at = {"layer_no": layer_num + 1, "oper_num": oper_num + 1}

        self.assertGreater(condition_at["layer_no"], measure_at["layer_no"])
