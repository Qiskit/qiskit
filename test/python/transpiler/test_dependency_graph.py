# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests the DependencyGraph."""
import unittest

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.transpiler.passes.mapping.algorithm import DependencyGraph


class TestDependencyGraph(QiskitTestCase):
    """Tests the DependencyGraph."""

    def test_qargs(self):
        """ Test for qargs.
        """
        qr = QuantumRegister(2, 'q')
        cr = ClassicalRegister(2, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[1], qr[0])
        circuit.measure(qr[0], cr[1])
        dep_graph = DependencyGraph(circuit)
        self.assertEqual(dep_graph.qargs(0), [(qr, 1), (qr, 0)])
        self.assertEqual(dep_graph.qargs(1), [(qr, 0)])

    def test_head_gates(self):
        """ Test for head gates.
        """
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[1], qr[2])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        dep_graph = DependencyGraph(circuit)
        self.assertEqual(dep_graph.head_gates(), set([0, 1]))

    def test_gr_successors(self):
        """ Test for successor gates of Gr (transitive reduction).
        """
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[1], qr[2])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        dep_graph = DependencyGraph(circuit)
        self.assertEqual(list(dep_graph.gr_successors(0)), [2, 3])
        self.assertEqual(list(dep_graph.gr_successors(2)), [4, 5])
        self.assertEqual(list(dep_graph.gr_successors(4)), [])

    def test_barrier(self):
        """ Test for barriers.
        """
        qr = QuantumRegister(4, 'b')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[0], qr[1])
        circuit.barrier(qr)
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[2], qr[3])
        dep_graph = DependencyGraph(circuit)
        expected = sorted([(0, 1), (1, 2), (1, 3)])
        self.assertEqual(list(dep_graph._graph.edges()), expected)

    def test_measure(self):
        """ Test for measurements.
        """
        qr = QuantumRegister(3, 'b')
        cr = ClassicalRegister(3, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.measure(qr[0], cr[1])
        circuit.cx(qr[1], qr[2])
        circuit.measure(qr[2], cr[1])  # overwrite -> introduce dependency (2, 4)
        dep_graph = DependencyGraph(circuit, graph_type="xz_commute")
        expected_edges = sorted([(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)])
        self.assertEqual(list(dep_graph._graph.edges()), expected_edges)

    def test_h_gate_in_the_middle(self):
        """ Test for a h gate in the middle of a quantum circuit.
        """
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[1], qr[2])
        circuit.h(qr[1])
        circuit.cx(qr[1], qr[0])
        dep_graph = DependencyGraph(circuit, graph_type="xz_commute")
        expected = sorted([(0, 2), (1, 2), (2, 3), (3, 4)])
        self.assertEqual(list(dep_graph._graph.edges()), expected)

    def test_rz_gate_in_the_middle(self):
        """ Test for a rz gate in the middle of a quantum circuit.
        """
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[1], qr[2])
        circuit.rz(1.0, qr[1])
        circuit.cx(qr[1], qr[0])
        dep_graph = DependencyGraph(circuit, graph_type="xz_commute")
        expected = sorted([(0, 2), (1, 2), (0, 3), (0, 4)])
        self.assertEqual(list(dep_graph._graph.edges()), expected)

    def test_rz_gate_in_the_middle_basic_graph_type(self):
        """ Test for a h gate in the middle of a quantum circuit with "basic" graph_type.
        """
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[1], qr[2])
        circuit.rz(1.0, qr[1])
        circuit.cx(qr[1], qr[0])
        dep_graph = DependencyGraph(circuit, graph_type="basic")
        expected = sorted([(0, 2), (1, 2), (2, 3), (3, 4)])
        self.assertEqual(list(dep_graph._graph.edges()), expected)


if __name__ == "__main__":
    unittest.main()
