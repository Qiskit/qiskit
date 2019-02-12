# -*- coding: utf-8 -*-
import unittest

# from qiskit.transpiler.passes.algorithm import DependencyGraph
from new_swappers.algorithm import DependencyGraph
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.test import QiskitTestCase


class TestDependencyGraph(QiskitTestCase):
    """Tests the DependencyGraph."""

    def test_qargs(self):
        q = QuantumRegister(2, 'q')
        c = ClassicalRegister(2, 'c')
        circuit = QuantumCircuit(q, c)
        circuit.cx(q[1], q[0])
        circuit.measure(q[0], c[1])
        dg = DependencyGraph(circuit)
        self.assertEqual(dg.qargs(0), [(q, 1), (q, 0)])
        self.assertEqual(dg.qargs(1), [(q, 0)])

    def test_head_gates(self):
        q = QuantumRegister(4, 'q')
        c = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(q, c)
        circuit.cx(q[0], q[1])
        circuit.cx(q[2], q[3])
        circuit.cx(q[1], q[2])
        circuit.h(q[0])
        circuit.cx(q[0], q[1])
        circuit.cx(q[2], q[3])
        dg = DependencyGraph(circuit)
        self.assertEqual(dg.head_gates(), set([0, 1]))

    def test_gr_successors(self):
        q = QuantumRegister(4, 'q')
        c = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(q, c)
        circuit.cx(q[0], q[1])
        circuit.cx(q[2], q[3])
        circuit.cx(q[1], q[2])
        circuit.h(q[0])
        circuit.cx(q[0], q[1])
        circuit.cx(q[2], q[3])
        dg = DependencyGraph(circuit)
        self.assertEqual(list(dg.gr_successors(0)), [2, 3])
        self.assertEqual(list(dg.gr_successors(2)), [4, 5])
        self.assertEqual(list(dg.gr_successors(4)), [])

    def test_barrier(self):
        b = QuantumRegister(4, 'b')
        c = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(b, c)
        circuit.cx(b[0], b[1])
        circuit.barrier(b)
        circuit.cx(b[1], b[0])
        circuit.cx(b[2], b[3])
        dg = DependencyGraph(circuit)
        expected = sorted([(0, 1), (1, 2), (1, 3)])
        self.assertEqual(list(dg.G.edges()), expected)

    def test_measure(self):
        b = QuantumRegister(3, 'b')
        c = ClassicalRegister(3, 'c')
        circuit = QuantumCircuit(b, c)
        circuit.h(b[0])
        circuit.cx(b[0], b[1])
        circuit.measure(b[0], c[1])
        circuit.cx(b[1], b[2])
        circuit.measure(b[2], c[1])  # overwrite -> introduce dependency (2, 4)
        dg = DependencyGraph(circuit, graph_type="xz_commute")
        expected_edges = sorted([(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)])
        self.assertEqual(list(dg.G.edges()), expected_edges)

    def test_h_gate_in_the_middle(self):
        q = QuantumRegister(4, 'q')
        c = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(q, c)
        circuit.cx(q[0], q[1])
        circuit.cx(q[2], q[3])
        circuit.cx(q[1], q[2])
        circuit.h(q[1])
        circuit.cx(q[1], q[0])
        dg = DependencyGraph(circuit, graph_type="xz_commute")
        expected = sorted([(0, 2), (1, 2), (2, 3), (3, 4)])
        self.assertEqual(list(dg.G.edges()), expected)

    def test_rz_gate_in_the_middle(self):
        q = QuantumRegister(4, 'q')
        c = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(q, c)
        circuit.cx(q[0], q[1])
        circuit.cx(q[2], q[3])
        circuit.cx(q[1], q[2])
        circuit.rz(1.0, q[1])
        circuit.cx(q[1], q[0])
        dg = DependencyGraph(circuit, graph_type="xz_commute")
        expected = sorted([(0, 2), (1, 2), (0, 3), (0, 4)])
        self.assertEqual(list(dg.G.edges()), expected)

    def test_rz_gate_in_the_middle_blg(self):
        q = QuantumRegister(4, 'q')
        c = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(q, c)
        circuit.cx(q[0], q[1])
        circuit.cx(q[2], q[3])
        circuit.cx(q[1], q[2])
        circuit.rz(1.0, q[1])
        circuit.cx(q[1], q[0])
        dg = DependencyGraph(circuit, graph_type="basic")
        expected = sorted([(0, 2), (1, 2), (2, 3), (3, 4)])
        self.assertEqual(list(dg.G.edges()), expected)


if __name__ == "__main__":
    unittest.main()
