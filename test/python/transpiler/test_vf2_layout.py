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

"""Test the VF2Layout pass"""

import unittest
import numpy
import retworkx

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import CouplingMap, Layout, Target, TranspilerError
from qiskit.transpiler.passes.layout.vf2_layout import VF2Layout, VF2LayoutStopReason
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeTenerife, FakeRueschlikon, FakeManhattan, FakeYorktown
from qiskit.circuit.library import GraphState, CXGate


class LayoutTestCase(QiskitTestCase):
    """VF2Layout assertions"""

    seed = 42

    def assertLayout(self, dag, coupling_map, property_set, strict_direction=False):
        """Checks if the circuit in dag was a perfect layout in property_set for the given
        coupling_map"""
        self.assertEqual(property_set["VF2Layout_stop_reason"], VF2LayoutStopReason.SOLUTION_FOUND)

        layout = property_set["layout"]
        edges = coupling_map.graph.edge_list()
        for gate in dag.two_qubit_ops():
            if dag.has_calibration_for(gate):
                continue
            physical_q0 = layout[gate.qargs[0]]
            physical_q1 = layout[gate.qargs[1]]

            if strict_direction:
                result = (physical_q0, physical_q1) in edges
            else:
                result = (physical_q0, physical_q1) in edges or (physical_q1, physical_q0) in edges
            self.assertTrue(result)


class TestVF2LayoutSimple(LayoutTestCase):
    """Tests the VF2Layout pass"""

    def test_2q_circuit_2q_coupling(self):
        """A simple example, without considering the direction
          0 - 1
        qr1 - qr0
        """
        cmap = CouplingMap([[0, 1]])

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap, strict_direction=False, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, cmap, pass_.property_set)

    def test_2q_circuit_2q_coupling_sd(self):
        """A simple example, considering the direction
         0  -> 1
        qr1 -> qr0
        """
        cmap = CouplingMap([[0, 1]])

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap, strict_direction=True, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, cmap, pass_.property_set, strict_direction=True)

    def test_3q_circuit_3q_coupling_non_induced(self):
        """A simple example, check for non-induced subgraph
            1         qr0 -> qr1 -> qr2
           / \
          0 - 2
        """
        cmap = CouplingMap([[0, 1], [1, 2], [2, 0]])

        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])  # qr0-> qr1
        circuit.cx(qr[1], qr[2])  # qr1-> qr2

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap, seed=-1, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, cmap, pass_.property_set)

    def test_call_limit(self):
        """Test that call limit is enforce."""
        cmap = CouplingMap([[0, 1], [1, 2], [2, 0]])

        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])  # qr0-> qr1
        circuit.cx(qr[1], qr[2])  # qr1-> qr2

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap, seed=-1, call_limit=1)
        pass_.run(dag)
        self.assertEqual(
            pass_.property_set["VF2Layout_stop_reason"], VF2LayoutStopReason.NO_SOLUTION_FOUND
        )

    def test_coupling_map_and_target(self):
        """Test that a Target is used instead of a CouplingMap if both are specified."""
        cmap = CouplingMap([[0, 1], [1, 2]])
        target = Target()
        target.add_instruction(CXGate(), {(0, 1): None, (1, 2): None, (1, 0): None})
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])  # qr0-> qr1
        circuit.cx(qr[1], qr[2])  # qr1-> qr2
        circuit.cx(qr[1], qr[0])  # qr1-> qr0
        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap, seed=-1, max_trials=1, target=target)
        pass_.run(dag)
        self.assertLayout(dag, target.build_coupling_map(), pass_.property_set)

    def test_neither_coupling_map_or_target(self):
        """Test that we raise if neither a target or coupling map is specified."""
        vf2_pass = VF2Layout(seed=123, call_limit=1000, time_limit=20, max_trials=7)
        circuit = QuantumCircuit(2)
        dag = circuit_to_dag(circuit)
        with self.assertRaises(TranspilerError):
            vf2_pass.run(dag)


class TestVF2LayoutLattice(LayoutTestCase):
    """Fit in 25x25 hexagonal lattice coupling map"""

    cmap25 = CouplingMap.from_hexagonal_lattice(25, 25, bidirectional=False)

    def graph_state_from_pygraph(self, graph):
        """Creates a GraphState circuit from a PyGraph"""
        adjacency_matrix = retworkx.adjacency_matrix(graph)
        return GraphState(adjacency_matrix).decompose()

    def test_hexagonal_lattice_graph_20_in_25(self):
        """A 20x20 interaction map in 25x25 coupling map"""
        graph_20_20 = retworkx.generators.hexagonal_lattice_graph(20, 20)
        circuit = self.graph_state_from_pygraph(graph_20_20)

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(self.cmap25, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, self.cmap25, pass_.property_set)

    def test_hexagonal_lattice_graph_9_in_25(self):
        """A 9x9 interaction map in 25x25 coupling map"""
        graph_9_9 = retworkx.generators.hexagonal_lattice_graph(9, 9)
        circuit = self.graph_state_from_pygraph(graph_9_9)

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(self.cmap25, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, self.cmap25, pass_.property_set)


class TestVF2LayoutBackend(LayoutTestCase):
    """Tests VF2Layout against backends"""

    def test_5q_circuit_Rueschlikon_no_solution(self):
        """5 qubits in Rueschlikon, no solution

        q0[1] ↖     ↗ q0[2]
               q0[0]
        q0[3] ↙     ↘ q0[4]
        """
        cmap16 = FakeRueschlikon().configuration().coupling_map

        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[0], qr[4])
        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(CouplingMap(cmap16), seed=self.seed, max_trials=1)
        pass_.run(dag)
        layout = pass_.property_set["layout"]
        self.assertIsNone(layout)
        self.assertEqual(
            pass_.property_set["VF2Layout_stop_reason"], VF2LayoutStopReason.NO_SOLUTION_FOUND
        )

    def test_9q_circuit_Rueschlikon_sd(self):
        """9 qubits in Rueschlikon, considering the direction

        1 →  2 →  3 →  4 ←  5 ←  6 →  7 ← 8
        ↓    ↑    ↓    ↓    ↑    ↓    ↓   ↑
        0 ← 15 → 14 ← 13 ← 12 → 11 → 10 ← 9
        """
        cmap16 = CouplingMap(FakeRueschlikon().configuration().coupling_map)

        qr0 = QuantumRegister(4, "q0")
        qr1 = QuantumRegister(5, "q1")
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr0[1], qr0[2])  # q0[1] -> q0[2]
        circuit.cx(qr0[0], qr1[3])  # q0[0] -> q1[3]
        circuit.cx(qr1[4], qr0[2])  # q1[4] -> q0[2]

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap16, strict_direction=True, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, cmap16, pass_.property_set)

    def test_4q_circuit_Tenerife_loose_nodes(self):
        """4 qubits in Tenerife, with loose nodes

            1
          ↙ ↑
        0 ← 2 ← 3
            ↑ ↙
            4
        """
        cmap5 = CouplingMap(FakeTenerife().configuration().coupling_map)

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        circuit.cx(qr[0], qr[2])  # qr0 -> qr2

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap5, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, cmap5, pass_.property_set)

    def test_3q_circuit_Tenerife_sd(self):
        """3 qubits in Tenerife, considering the direction
            1                       1
          ↙ ↑                    ↙  ↑
        0 ← 2 ← 3              0 ← qr2 ← qr1
            ↑ ↙                     ↑  ↙
            4                      qr0
        """
        cmap5 = CouplingMap(FakeTenerife().configuration().coupling_map)

        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        circuit.cx(qr[0], qr[2])  # qr0 -> qr2
        circuit.cx(qr[1], qr[2])  # qr1 -> qr2

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap5, strict_direction=True, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, cmap5, pass_.property_set, strict_direction=True)

    def test_9q_circuit_Rueschlikon(self):
        """9 qubits in Rueschlikon, without considering the direction

        1 →  2 →  3 →  4 ←  5 ←  6 →  7 ← 8
        ↓    ↑    ↓    ↓    ↑    ↓    ↓   ↑
        0 ← 15 → 14 ← 13 ← 12 → 11 → 10 ← 9

          1 -- q1_0 - q1_1 - 4 --- 5 --  6  - 7 --- q0_1
          |    |      |      |     |     |    |      |
        q1_2 - q1_3 - q0_0 - 13 - q0_3 - 11 - q1_4 - q0_2
        """
        cmap16 = CouplingMap(FakeRueschlikon().configuration().coupling_map)

        qr0 = QuantumRegister(4, "q0")
        qr1 = QuantumRegister(5, "q1")
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr0[1], qr0[2])  # q0[1] -> q0[2]
        circuit.cx(qr0[0], qr1[3])  # q0[0] -> q1[3]
        circuit.cx(qr1[4], qr0[2])  # q1[4] -> q0[2]

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap16, strict_direction=False, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, cmap16, pass_.property_set)

    def test_3q_circuit_Tenerife(self):
        """3 qubits in Tenerife, without considering the direction

            1                    1
          ↙ ↑                 /  |
        0 ← 2 ← 3           0 - qr1 - qr2
            ↑ ↙                 |   /
            4                   qr0
        """
        cmap5 = CouplingMap(FakeTenerife().configuration().coupling_map)

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        circuit.cx(qr[0], qr[2])  # qr0 -> qr2
        circuit.cx(qr[1], qr[2])  # qr1 -> qr2

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap5, strict_direction=False, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, cmap5, pass_.property_set)

    def test_perfect_fit_Manhattan(self):
        """A circuit that fits perfectly in Manhattan (65 qubits)
        See https://github.com/Qiskit/qiskit-terra/issues/5694"""
        manhattan_cm = FakeManhattan().configuration().coupling_map
        cmap65 = CouplingMap(manhattan_cm)

        rows = [x[0] for x in manhattan_cm]
        cols = [x[1] for x in manhattan_cm]

        adj_matrix = numpy.zeros((65, 65))
        adj_matrix[rows, cols] = 1

        circuit = GraphState(adj_matrix).decompose()
        circuit.measure_all()

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap65, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, cmap65, pass_.property_set)


class TestVF2LayoutOther(LayoutTestCase):
    """Other VF2Layout tests"""

    def test_seed(self):
        """Different seeds yield different results"""
        seed_1 = 42
        seed_2 = 45

        cmap5 = FakeTenerife().configuration().coupling_map

        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        circuit.cx(qr[0], qr[2])  # qr0 -> qr2
        circuit.cx(qr[1], qr[2])  # qr1 -> qr2
        dag = circuit_to_dag(circuit)

        pass_1 = VF2Layout(CouplingMap(cmap5), seed=seed_1, max_trials=1)
        pass_1.run(dag)
        layout_1 = pass_1.property_set["layout"]
        self.assertEqual(
            pass_1.property_set["VF2Layout_stop_reason"], VF2LayoutStopReason.SOLUTION_FOUND
        )

        pass_2 = VF2Layout(CouplingMap(cmap5), seed=seed_2, max_trials=1)
        pass_2.run(dag)
        layout_2 = pass_2.property_set["layout"]
        self.assertEqual(
            pass_2.property_set["VF2Layout_stop_reason"], VF2LayoutStopReason.SOLUTION_FOUND
        )

        self.assertNotEqual(layout_1, layout_2)

    def test_3_q_gate(self):
        """The pass does not handle gates with more than 2 qubits"""
        seed_1 = 42

        cmap5 = FakeTenerife().configuration().coupling_map

        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.ccx(qr[1], qr[0], qr[2])
        dag = circuit_to_dag(circuit)

        pass_1 = VF2Layout(CouplingMap(cmap5), seed=seed_1, max_trials=1)
        pass_1.run(dag)
        self.assertEqual(
            pass_1.property_set["VF2Layout_stop_reason"], VF2LayoutStopReason.MORE_THAN_2Q
        )


class TestScoreHeuristic(QiskitTestCase):
    """Test the internal score heuristic of the pass."""

    def test_no_properties(self):
        """Test scores with no properties."""
        vf2_pass = VF2Layout(
            CouplingMap(
                [
                    (0, 1),
                    (0, 2),
                    (0, 3),
                    (1, 0),
                    (1, 2),
                    (1, 3),
                    (2, 0),
                    (2, 1),
                    (2, 2),
                    (2, 3),
                    (3, 0),
                    (3, 1),
                    (3, 2),
                    (4, 0),
                    (0, 4),
                    (5, 1),
                    (1, 5),
                ]
            )
        )
        qr = QuantumRegister(2)
        layout = Layout({qr[0]: 0, qr[1]: 1})
        score = vf2_pass._score_layout(layout)
        self.assertEqual(score, 16)
        better_layout = Layout({qr[0]: 4, qr[1]: 5})
        better_score = vf2_pass._score_layout(better_layout)
        self.assertEqual(4, better_score)

    def test_with_properties(self):
        """Test scores with properties."""
        backend = FakeYorktown()
        cmap = CouplingMap(backend.configuration().coupling_map)
        properties = backend.properties()
        vf2_pass = VF2Layout(cmap, properties=properties)
        qr = QuantumRegister(2)
        layout = Layout({qr[0]: 4, qr[1]: 2})
        bad_score = vf2_pass._score_layout(layout)
        self.assertAlmostEqual(0.4075, bad_score)
        better_layout = Layout({qr[0]: 1, qr[1]: 3})
        better_score = vf2_pass._score_layout(better_layout)
        self.assertAlmostEqual(0.0588, better_score)


class TestMultipleTrials(QiskitTestCase):
    """Test the passes behavior with >1 trial."""

    def test_no_properties(self):
        """Test it finds the lowest degree perfect layout with no properties."""
        vf2_pass = VF2Layout(
            CouplingMap(
                [
                    (0, 1),
                    (0, 2),
                    (0, 3),
                    (1, 0),
                    (1, 2),
                    (1, 3),
                    (2, 0),
                    (2, 1),
                    (2, 2),
                    (2, 3),
                    (3, 0),
                    (3, 1),
                    (3, 2),
                    (4, 0),
                    (0, 4),
                    (5, 1),
                    (1, 5),
                ]
            )
        )
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        qc.x(qr)
        qc.measure_all()
        property_set = {}
        vf2_pass(qc, property_set)
        self.assertEqual(set(property_set["layout"].get_physical_bits()), {5, 4})

    def test_with_properties(self):
        """Test it finds the least noise perfect layout with no properties."""
        backend = FakeYorktown()
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        qc.x(qr)
        qc.measure_all()
        cmap = CouplingMap(backend.configuration().coupling_map)
        properties = backend.properties()
        vf2_pass = VF2Layout(cmap, properties=properties)
        property_set = {}
        vf2_pass(qc, property_set)
        self.assertEqual(set(property_set["layout"].get_physical_bits()), {1, 3})

    def test_max_trials_exceeded(self):
        """Test it exits when max_trials is reached."""
        backend = FakeYorktown()
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        qc.x(qr)
        qc.measure_all()
        cmap = CouplingMap(backend.configuration().coupling_map)
        properties = backend.properties()
        vf2_pass = VF2Layout(cmap, properties=properties, seed=-1, max_trials=1)
        property_set = {}
        with self.assertLogs("qiskit.transpiler.passes.layout.vf2_layout", level="DEBUG") as cm:
            vf2_pass(qc, property_set)
        self.assertIn(
            "DEBUG:qiskit.transpiler.passes.layout.vf2_layout:Trial 1 is >= configured max trials 1",
            cm.output,
        )
        self.assertEqual(set(property_set["layout"].get_physical_bits()), {2, 0})

    def test_time_limit_exceeded(self):
        """Test the pass stops after time_limit is reached."""
        backend = FakeYorktown()
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        qc.x(qr)
        qc.measure_all()
        cmap = CouplingMap(backend.configuration().coupling_map)
        properties = backend.properties()
        vf2_pass = VF2Layout(cmap, properties=properties, seed=-1, time_limit=0.0)
        property_set = {}
        with self.assertLogs("qiskit.transpiler.passes.layout.vf2_layout", level="DEBUG") as cm:
            vf2_pass(qc, property_set)
        for output in cm.output:
            if output.startswith(
                "DEBUG:qiskit.transpiler.passes.layout.vf2_layout:VF2Layout has taken"
            ) and output.endswith("which exceeds configured max time: 0.0"):
                break
        else:
            self.fail("No failure debug log message found")

        self.assertEqual(set(property_set["layout"].get_physical_bits()), {2, 0})

    def test_reasonable_limits_for_simple_layouts(self):
        """Test that the default trials is set to a reasonable number."""
        backend = FakeManhattan()
        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        cmap = CouplingMap(backend.configuration().coupling_map)
        properties = backend.properties()
        # Run without any limits set
        vf2_pass = VF2Layout(cmap, properties=properties, seed=42)
        property_set = {}
        with self.assertLogs("qiskit.transpiler.passes.layout.vf2_layout", level="DEBUG") as cm:
            vf2_pass(qc, property_set)
        self.assertIn(
            "DEBUG:qiskit.transpiler.passes.layout.vf2_layout:Trial 159 is >= configured max trials 159",
            cm.output,
        )
        self.assertEqual(set(property_set["layout"].get_physical_bits()), {48, 49, 40, 47, 58})

    def test_no_limits_with_negative(self):
        """Test that we're not enforcing a trial limit if set to negative."""
        backend = FakeYorktown()
        qc = QuantumCircuit(3)
        qc.h(0)
        cmap = CouplingMap(backend.configuration().coupling_map)
        implicit_max = len(cmap.graph.edge_list()) + 15
        properties = backend.properties()
        # Run without any limits set
        vf2_pass = VF2Layout(cmap, properties=properties, seed=42, max_trials=0)
        property_set = {}
        with self.assertLogs("qiskit.transpiler.passes.layout.vf2_layout", level="DEBUG") as cm:
            vf2_pass(qc, property_set)
        for output in cm.output:
            self.assertNotIn("is >= configured max trials", output)
        last_line = cm.output[-1]
        # The last line should be
        # DEBUG:qiskit.transpiler.passes.layout.vf2_layout: Trial n has score 0.122
        trials = int(last_line.split(" ")[1])
        self.assertGreater(trials, implicit_max)
        self.assertEqual(set(property_set["layout"].get_physical_bits()), {3, 1, 0})


if __name__ == "__main__":
    unittest.main()
