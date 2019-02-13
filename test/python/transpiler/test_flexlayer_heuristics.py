# -*- coding: utf-8 -*-
import unittest

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.mapper import CouplingMap, Layout
from qiskit.transpiler.passes.mapping.algorithm import DependencyGraph
from qiskit.transpiler.passes.mapping.algorithm import FlexlayerHeuristics, remove_head_swaps


class TestLookaheadHeuristics(unittest.TestCase):
    """Tests for dependency_graph.py"""

    @unittest.skip("due to a bug in DAGCircuit.__eq__()")
    def test_search_n4cx4h1(self):
        q = QuantumRegister(4, 'q')
        c = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(q, c)
        circuit.cx(q[0], q[1])
        circuit.cx(q[2], q[3])
        circuit.cx(q[1], q[2])
        circuit.h(q[1])
        circuit.cx(q[1], q[0])
        circuit.barrier()
        for i in range(4):
            circuit.measure(q[i], c[i])
        dg = DependencyGraph(circuit, graph_type="basic")
        coupling = CouplingMap([[0, 1], [1, 2], [1, 3]])  # {0: [1], 1: [2, 3]}
        initial_layout = Layout.generate_trivial_layout(q)
        algo = FlexlayerHeuristics(circuit, dg, coupling, initial_layout)
        actual_dag, layout = algo.search()

        expected = QuantumCircuit(q, c)
        expected.cx(q[0], q[1])
        expected.swap(q[1], q[2])
        expected.cx(q[1], q[3])
        expected.cx(q[2], q[1])
        expected.h(q[2])
        expected.swap(q[0], q[1])
        expected.cx(q[2], q[1])
        expected.barrier()
        expected.measure(q[1], c[0])
        expected.measure(q[2], c[1])
        expected.measure(q[0], c[2])
        expected.measure(q[3], c[3])
        expected_layout = initial_layout
        from qiskit.tools.visualization import dag_drawer
        dag_drawer(actual_dag)
        dag_drawer(circuit_to_dag(expected))
        print(dag_to_circuit(actual_dag).draw())
        print(expected.draw())
        self.assertEqual(actual_dag, circuit_to_dag(expected))
        self.assertEqual(layout, expected_layout)

    @unittest.skip("TODO: Change to use DAGCircuit and Layout")
    def test_search_multi_creg(self):
        b = QuantumRegister(4, 'b')
        c = ClassicalRegister(2, 'c')
        d = ClassicalRegister(2, 'd')
        circuit = QuantumCircuit(b, c, d)
        circuit.cx(b[0], b[1])
        circuit.cx(b[2], b[3])
        circuit.cx(b[1], b[2])
        circuit.h(b[1])
        circuit.cx(b[1], b[0])
        circuit.barrier(b)
        circuit.measure(b[0], c[0])
        circuit.measure(b[1], c[1])
        circuit.measure(b[2], d[0])
        circuit.measure(b[3], d[1])
        dg = DependencyGraph(circuit, graph_type="basic")
        coupling = CouplingMap([[0, 1], [1, 2], [1, 3]])  # {0: [1], 1: [2, 3]}
        initial_layout = Layout.generate_trivial_layout(b)
        algo = FlexlayerHeuristics(circuit, dg, coupling, initial_layout)
        qc, layout = algo.search()
        actual_measures = [s for s in qc.qasm().split('\n') if s.startswith("measure")]
        expected_measures = []
        expected_measures.append("measure q[0] -> d[0];")
        expected_measures.append("measure q[1] -> c[0];")
        expected_measures.append("measure q[2] -> c[1];")
        expected_measures.append("measure q[3] -> d[1];")
        expected_layout = initial_layout
        self.assertEqual(sorted(actual_measures), expected_measures)
        self.assertEqual(layout, expected_layout)

    @unittest.skip("TODO: Change to use DAGCircuit and Layout")
    def test_remove_head_swaps(self):
        q = QuantumRegister(4, 'q')
        c = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(q, c)
        circuit.cx(q[0], q[1])
        circuit.u1(1, q[2])
        circuit.swap(q[2], q[3])
        circuit.cx(q[1], q[2])
        for i in range(4):
            circuit.measure(q[i], c[i])
        initial_layout = {('q', i): ('q', i) for i in range(4)}
        resqc, layout = remove_head_swaps(circuit, initial_layout)
        actual_qasm = ''.join([s for s in resqc.qasm().split('\n') if not s.startswith("measure")])
        actual_measure = sorted([s for s in resqc.qasm().split('\n') if s.startswith("measure")])
        header = 'OPENQASM 2.0;include "qelib1.inc";qreg q[4];creg c[4];'
        expected_qasm = header + "cx q[0],q[1];u1(1) q[3];cx q[1],q[2];"
        expected_measure = ["measure q[%d] -> c[%d];" % (i, i) for i in range(4)]
        expected_layout = {('q', 0): ('q', 0), ('q', 1): ('q', 1), ('q', 3): ('q', 2),
                           ('q', 2): ('q', 3)}
        self.assertEqual(actual_qasm, expected_qasm)
        self.assertEqual(actual_measure, expected_measure)
        self.assertEqual(layout, expected_layout)

    # @unittest.skip("WIP")
    # def test_search_rd32_v0_67(self):
    #     qp = QuantumProgram()
    #     qasm_text = 'OPENQASM 2.0;\
    #         include "qelib1.inc";\
    #         gate peres a,b,c {ccx a, b, c; cx a, b;}\
    #         qreg q[4];\
    #         creg c[4];\
    #         peres q[0], q[1], q[3];\
    #         peres q[1], q[2], q[3];\
    #         barrier q;\
    #         measure q->c;'
    #     qc = load_qasm_string(qasm_text, basis_gates="u1,u2,u3,cx,id")
    #     dg = DependencyGraph(qc, graph_type="basic")
    #     coupling = Coupling({1: [0], 2: [0, 1, 4], 3: [2, 4]})
    #     initial_layout = {('q', 0): ('q', 1), ('q', 1): ('q', 0), ('q', 2): ('q', 2),
    #                       ('q', 3): ('q', 4)}
    #     algo = FlexlayerHeuristics(qc, dg, coupling, initial_layout)
    #     dag, layout = algo.search()

    @unittest.skip("TODO: Change to use DAGCircuit and Layout")
    def test_remove_head_swaps2(self):
        q = QuantumRegister(6, 'q')
        c = ClassicalRegister(6, 'c')
        circuit = QuantumCircuit(q, c)
        circuit.u1(1, q[4])
        circuit.h(q[3])
        circuit.swap(q[4], q[5])
        circuit.cx(q[3], q[4])
        circuit.swap(q[3], q[4])
        circuit.u1(1, q[1])
        circuit.h(q[0])
        circuit.swap(q[0], q[1])
        circuit.swap(q[1], q[2])
        circuit.swap(q[2], q[3])
        circuit.cx(q[3], q[4])
        initial_layout = {('b', i): ('q', i) for i in range(6)}
        resqc, layout = remove_head_swaps(circuit, initial_layout)
        actual_qasm = ''.join([s for s in resqc.qasm().split('\n')])
        header = 'OPENQASM 2.0;include "qelib1.inc";qreg q[6];creg c[6];'
        expected_qasm = header + "u1(1) q[5];h q[3];cx q[3],q[4];swap q[3],q[4];u1(1) q[0];" \
                                 "h q[2];swap q[2],q[3];cx q[3],q[4];"
        expected_layout = {('b', 0): ('q', 2), ('b', 1): ('q', 0), ('b', 2): ('q', 1),
                           ('b', 3): ('q', 3), ('b', 4): ('q', 5), ('b', 5): ('q', 4)}
        self.assertEqual(actual_qasm, expected_qasm)
        self.assertEqual(layout, expected_layout)


if __name__ == "__main__":
    unittest.main()
