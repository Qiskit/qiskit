# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for FlexlayerHeuristics."""

import unittest

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.mapper import CouplingMap, Layout
from qiskit.transpiler.passes.mapping.algorithm import DependencyGraph
from qiskit.transpiler.passes.mapping.algorithm import FlexlayerHeuristics, remove_head_swaps


class TestFlexlayerHeuristics(unittest.TestCase):
    """Tests for FlexlayerHeuristics."""

    @unittest.skip("due to a bug in DAGCircuit.__eq__()")
    def test_search_4qcx4h1(self):
        """Test for 4 cx gates and 1 h gate in a 4q circuit.
        """
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[1], qr[2])
        circuit.h(qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.barrier()
        for i in range(4):
            circuit.measure(qr[i], cr[i])
        dep_graph = DependencyGraph(circuit, graph_type="basic")
        coupling = CouplingMap([[0, 1], [1, 2], [1, 3]])  # {0: [1], 1: [2, 3]}
        initial_layout = Layout.generate_trivial_layout(qr)
        algo = FlexlayerHeuristics(circuit, dep_graph, coupling, initial_layout)
        actual_dag, layout = algo.search()

        expected = QuantumCircuit(qr, cr)
        expected.cx(qr[0], qr[1])
        expected.swap(qr[1], qr[2])
        expected.cx(qr[1], qr[3])
        expected.cx(qr[2], qr[1])
        expected.h(qr[2])
        expected.swap(qr[0], qr[1])
        expected.cx(qr[2], qr[1])
        expected.barrier()
        expected.measure(qr[1], cr[0])
        expected.measure(qr[2], cr[1])
        expected.measure(qr[0], cr[2])
        expected.measure(qr[3], cr[3])
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
        """Test for multiple ClassicalRegisters.
        """
        qr = QuantumRegister(4, 'b')
        cr1 = ClassicalRegister(2, 'c')
        cr2 = ClassicalRegister(2, 'd')
        circuit = QuantumCircuit(qr, cr1, cr2)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[1], qr[2])
        circuit.h(qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.barrier(qr)
        circuit.measure(qr[0], cr1[0])
        circuit.measure(qr[1], cr1[1])
        circuit.measure(qr[2], cr2[0])
        circuit.measure(qr[3], cr2[1])
        dep_graph = DependencyGraph(circuit, graph_type="basic")
        coupling = CouplingMap([[0, 1], [1, 2], [1, 3]])  # {0: [1], 1: [2, 3]}
        initial_layout = Layout.generate_trivial_layout(qr)
        algo = FlexlayerHeuristics(circuit, dep_graph, coupling, initial_layout)
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
        """Test for removing unnecessary swap gates from qc by changing initial_layout.
        """
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[0], qr[1])
        circuit.u1(1, qr[2])
        circuit.swap(qr[2], qr[3])
        circuit.cx(qr[1], qr[2])
        for i in range(4):
            circuit.measure(qr[i], cr[i])
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
    def test_remove_head_swaps_and_remain_middle_swaps(self):
        """Test for only head swaps are removed, and middle swaps are remained properly.
        """
        qr = QuantumRegister(6, 'q')
        cr = ClassicalRegister(6, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.u1(1, qr[4])
        circuit.h(qr[3])
        circuit.swap(qr[4], qr[5])
        circuit.cx(qr[3], qr[4])
        circuit.swap(qr[3], qr[4])
        circuit.u1(1, qr[1])
        circuit.h(qr[0])
        circuit.swap(qr[0], qr[1])
        circuit.swap(qr[1], qr[2])
        circuit.swap(qr[2], qr[3])
        circuit.cx(qr[3], qr[4])
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
