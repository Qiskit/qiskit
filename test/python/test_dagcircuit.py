# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test for the DAGCircuit object"""

import unittest

from qiskit.dagcircuit import DAGCircuit
from qiskit._quantumregister import QuantumRegister
from qiskit._classicalregister import ClassicalRegister
from qiskit._quantumcircuit import QuantumCircuit
from .common import QiskitTestCase


class TestDagCircuit(QiskitTestCase):
    """Testing the dag circuit representation"""

    def test_create(self):
        """
        Creation using add_basis_element(), add_qreg(), add_creg(), and apply_operation_back()."""
        qreg = QuantumRegister(2, 'qr')
        creg = ClassicalRegister(2, 'cr')
        qubit0 = ('qr', 0)
        qubit1 = ('qr', 1)
        clbit0 = ('cr', 0)
        clbit1 = ('cr', 1)
        condition = ('cr', 3)
        dag = DAGCircuit()
        dag.add_basis_element('h', 1, number_classical=0, number_parameters=0)
        dag.add_basis_element('cx', 2)
        dag.add_basis_element('x', 1)
        dag.add_basis_element('measure', 1, number_classical=1, number_parameters=0)
        dag.add_qreg(qreg)
        dag.add_creg(creg)
        dag.apply_operation_back('h', [qubit0], [], [], condition=None)
        dag.apply_operation_back('cx', [qubit0, qubit1], [], [], condition=None)
        dag.apply_operation_back('measure', [qubit1], [clbit1], [], condition=None)
        dag.apply_operation_back('x', [qubit1], [], [], condition=condition)
        dag.apply_operation_back('measure', [qubit0], [clbit0], [], condition=None)
        dag.apply_operation_back('measure', [qubit1], [clbit1], [], condition=None)
        self.assertEqual(len(dag.multi_graph.nodes), 14)
        self.assertEqual(len(dag.multi_graph.edges), 16)

    def test_get_named_nodes(self):
        """ The get_named_nodes() method """
        qreg = QuantumRegister(3, 'q')
        dag = DAGCircuit()
        dag.add_basis_element('h', 1, number_classical=0, number_parameters=0)
        dag.add_basis_element('cx', 2)
        dag.add_qreg(qreg)
        dag.apply_operation_back('cx', [('q', 0), ('q', 1)])
        dag.apply_operation_back('h', [('q', 0)])
        dag.apply_operation_back('cx', [('q', 2), ('q', 1)])
        dag.apply_operation_back('cx', [('q', 0), ('q', 2)])
        dag.apply_operation_back('h', [('q', 2)])

        # The ordering is not assured, so we only compare the output (unordered) sets.
        # We use tuples because lists aren't hashable.
        named_nodes = dag.get_named_nodes('cx')
        node_qargs = {tuple(dag.multi_graph.node[node_id]["qargs"]) for node_id in named_nodes}
        expected_gates = {
            (('q', 0), ('q', 1)),
            (('q', 2), ('q', 1)),
            (('q', 0), ('q', 2))}
        self.assertEqual(expected_gates, node_qargs)

    def test_nodes_in_topological_order(self):
        """ The node_nums_in_topological_order() method"""
        dag = DAGCircuit()
        dag.add_basis_element('h', 1, number_classical=0, number_parameters=0)
        dag.add_basis_element('cx', 2)
        dag.add_qreg(QuantumRegister(3, "q"))
        dag.apply_operation_back('cx', [('q', 0), ('q', 1)])
        dag.apply_operation_back('h', [('q', 0)])
        dag.apply_operation_back('cx', [('q', 2), ('q', 1)])
        dag.apply_operation_back('cx', [('q', 0), ('q', 2)])
        dag.apply_operation_back('h', [('q', 2)])

        named_nodes = dag.node_nums_in_topological_order()
        self.assertEqual([5, 3, 1, 7, 9, 4, 8, 10, 11, 6, 2], [i for i in named_nodes])

    def test_layers_basic(self):
        """ The layers() method."""
        qreg = QuantumRegister(2, 'qr')
        creg = ClassicalRegister(2, 'cr')
        qubit0 = ('qr', 0)
        qubit1 = ('qr', 1)
        clbit0 = ('cr', 0)
        clbit1 = ('cr', 1)
        condition = ('cr', 3)
        dag = DAGCircuit()
        dag.add_basis_element('h', 1, number_classical=0, number_parameters=0)
        dag.add_basis_element('cx', 2)
        dag.add_basis_element('x', 1)
        dag.add_basis_element('measure', 1, number_classical=1, number_parameters=0)
        dag.add_qreg(qreg)
        dag.add_creg(creg)
        dag.apply_operation_back('h', [qubit0], [], [], condition=None)
        dag.apply_operation_back('cx', [qubit0, qubit1], [], [], condition=None)
        dag.apply_operation_back('measure', [qubit1], [clbit1], [], condition=None)
        dag.apply_operation_back('x', [qubit1], [], [], condition=condition)
        dag.apply_operation_back('measure', [qubit0], [clbit0], [], condition=None)
        dag.apply_operation_back('measure', [qubit1], [clbit1], [], condition=None)

        layers = list(dag.layers())
        self.assertEqual(5, len(layers))

        name_layers = [
            [node[1]["name"]
             for node in layer["graph"].multi_graph.nodes(data=True)
             if node[1]["type"] == "op"] for layer in layers]

        self.assertEqual([
            ['h'],
            ['cx'],
            ['measure'],
            ['x'],
            ['measure', 'measure']
        ], name_layers)


class TestCircuitProperties(QiskitTestCase):
    """DAGCircuit properties test."""

    def setUp(self):
        qr1 = QuantumRegister(4)
        qr2 = QuantumRegister(2)
        circ = QuantumCircuit(qr1, qr2)
        circ.h(qr1[0])
        circ.cx(qr1[2], qr1[3])
        circ.h(qr1[2])
        circ.t(qr1[2])
        circ.ch(qr1[2], qr1[1])
        circ.u2(0.1, 0.2, qr1[3])
        circ.ccx(qr2[0], qr2[1], qr1[0])

        self.dag = DAGCircuit.fromQuantumCircuit(circ)

    def test_circuit_size(self):
        """Test total number of operations in circuit."""
        self.assertEqual(self.dag.size(), 7)

    def test_circuit_depth(self):
        """Test circuit depth."""
        self.assertEqual(self.dag.depth(), 4)

    def test_circuit_width(self):
        """Test number of qubits in circuit."""
        self.assertEqual(self.dag.width(), 6)

    def test_circuit_operations(self):
        """Test circuit operations breakdown by kind of op."""
        operations = {
            'h': 2,
            't': 1,
            'u2': 1,
            'cx': 1,
            'ch': 1,
            'ccx': 1
        }

        self.assertDictEqual(self.dag.count_ops(), operations)

    def test_circuit_factors(self):
        """Test number of separable factors in circuit."""
        self.assertEqual(self.dag.num_tensor_factors(), 2)


class TestDagEquivalence(QiskitTestCase):
    """DAGCircuit equivalence check."""

    def setUp(self):
        self.qr1 = QuantumRegister(4, 'qr1')
        self.qr2 = QuantumRegister(2, 'qr2')
        circ1 = QuantumCircuit(self.qr2, self.qr1)
        circ1.h(self.qr1[0])
        circ1.cx(self.qr1[2], self.qr1[3])
        circ1.h(self.qr1[2])
        circ1.t(self.qr1[2])
        circ1.ch(self.qr1[2], self.qr1[1])
        circ1.u2(0.1, 0.2, self.qr1[3])
        circ1.ccx(self.qr2[0], self.qr2[1], self.qr1[0])
        self.dag1 = DAGCircuit.fromQuantumCircuit(circ1)

    def test_dag_eq(self):
        """ DAG equivalence check: True."""
        circ2 = QuantumCircuit(self.qr1, self.qr2)
        circ2.cx(self.qr1[2], self.qr1[3])
        circ2.u2(0.1, 0.2, self.qr1[3])
        circ2.h(self.qr1[0])
        circ2.h(self.qr1[2])
        circ2.t(self.qr1[2])
        circ2.ch(self.qr1[2], self.qr1[1])
        circ2.ccx(self.qr2[0], self.qr2[1], self.qr1[0])
        dag2 = DAGCircuit.fromQuantumCircuit(circ2)

        self.assertEqual(self.dag1, dag2)

    def test_dag_neq_topology(self):
        """ DAG equivalence check: False. Different topology."""
        circ2 = QuantumCircuit(self.qr1, self.qr2)
        circ2.cx(self.qr1[2], self.qr1[3])
        circ2.u2(0.1, 0.2, self.qr1[3])
        circ2.h(self.qr1[0])
        circ2.h(self.qr1[2])
        circ2.t(self.qr1[2])
        circ2.ch(self.qr1[0], self.qr1[1])  # <--- The difference: ch(qr1[2], qr1[1])
        circ2.ccx(self.qr2[0], self.qr2[1], self.qr1[0])
        dag2 = DAGCircuit.fromQuantumCircuit(circ2)

        self.assertNotEqual(self.dag1, dag2)

    def test_dag_neq_same_topology(self):
        """ DAG equivalence check: False. Same topology."""
        circ2 = QuantumCircuit(self.qr1, self.qr2)
        circ2.cx(self.qr1[2], self.qr1[3])
        circ2.u2(0.1, 0.2, self.qr1[3])
        circ2.h(self.qr1[0])
        circ2.h(self.qr1[2])
        circ2.t(self.qr1[2])
        circ2.cx(self.qr1[2], self.qr1[1])  # <--- The difference: ch(qr1[2], qr1[1])
        circ2.ccx(self.qr2[0], self.qr2[1], self.qr1[0])
        dag2 = DAGCircuit.fromQuantumCircuit(circ2)

        self.assertNotEqual(self.dag1, dag2)


if __name__ == '__main__':
    unittest.main()
