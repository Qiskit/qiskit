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
from qiskit._measure import Measure
from qiskit._reset import Reset
from qiskit._gate import Gate
from qiskit.extensions.standard.h import HGate
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.x import XGate
from qiskit.dagcircuit._dagcircuiterror import DAGCircuitError
from .common import QiskitTestCase


class TestDagRegisters(QiskitTestCase):
    """Test qreg and creg inside the dag"""

    def test_add_qreg_creg(self):
        dag = DAGCircuit()
        dag.add_qreg(QuantumRegister(2))
        dag.add_creg(ClassicalRegister(1))

    def test_add_reg_duplicate(self):
        dag = DAGCircuit()
        q = QuantumRegister(2)
        dag.add_qreg(q)
        self.assertRaises(DAGCircuitError, dag.add_qreg, q)

    def test_add_reg_duplicate_name(self):
        dag = DAGCircuit()
        q1 = QuantumRegister(3, 'q')
        dag.add_qreg(q1)
        q2 = QuantumRegister(2, 'q')
        self.assertRaises(DAGCircuitError, dag.add_qreg, q2)

    def test_add_reg_bad_type(self):
        dag = DAGCircuit()
        c = ClassicalRegister(2)
        self.assertRaises(DAGCircuitError, dag.add_qreg, c)

    def test_rename_register(self):
        dag = DAGCircuit()
        q = QuantumRegister(2, 'q')
        dag.add_qreg(q)
        dag.rename_register('q', 'v')
        self.assertTrue('v' in dag.qregs)
        self.assertEqual(dag.qregs['v'], q)


class TestDagOperations(QiskitTestCase):
    """Test ops inside the dag"""

    def setUp(self):
        self.dag = DAGCircuit()        
        qreg = QuantumRegister(3, 'qr')
        creg = ClassicalRegister(2, 'cr')
        self.dag.add_qreg(qreg)
        self.dag.add_creg(creg)
        self.dag.add_basis_element(name='h', number_qubits=1,
                                   number_classical=0, number_parameters=0)
        self.dag.add_basis_element('cx', 2, 0, 0)
        self.dag.add_basis_element('x', 1, 0, 0)
        self.dag.add_basis_element('measure', 1, 1, 0)
        self.dag.add_basis_element('reset', 1, 0, 0)

        self.qubit0 = qreg[0]
        self.qubit1 = qreg[1]
        self.qubit2 = qreg[2]
        self.clbit0 = creg[0]
        self.clbit1 = creg[1]
        self.condition = (creg, 3)

    def test_apply_operation_back(self):
        self.dag.apply_operation_back(HGate(self.qubit0), condition=None)
        self.dag.apply_operation_back(CnotGate(self.qubit0, self.qubit1), condition=None)
        self.dag.apply_operation_back(Measure(self.qubit1, self.clbit1), condition=None)
        self.dag.apply_operation_back(XGate(self.qubit1), condition=self.condition)
        self.dag.apply_operation_back(Measure(self.qubit0, self.clbit0), condition=None)
        self.dag.apply_operation_back(Measure(self.qubit1, self.clbit1), condition=None)
        self.assertEqual(len(self.dag.multi_graph.nodes), 16)
        self.assertEqual(len(self.dag.multi_graph.edges), 17)

    @unittest.expectedFailure
    # expected to fail until this function is fixed
    # see: https://github.com/Qiskit/qiskit-terra/issues/1235
    def test_apply_operation_front(self):
        self.dag.apply_operation_back(HGate(self.qubit0))
        self.dag.apply_operation_front(Reset(self.qubit0))
        h_node = self.dag.get_op_nodes(op=HGate(self.qubit0)).pop()
        reset_node = self.dag.get_op_nodes(op=Reset(self.qubit0)).pop()

        self.assertIn(reset_node, set(self.dag.multi_graph.predecessors(h_node)))

    def test_get_op_nodes_all(self):
        self.dag.apply_operation_back(HGate(self.qubit0))
        self.dag.apply_operation_back(CnotGate(self.qubit0, self.qubit1))

        op_nodes = self.dag.get_op_nodes()
        self.assertEqual(len(op_nodes), 2)

        op_node_1 = op_nodes.pop()
        op_node_2 = op_nodes.pop()
        self.assertIsInstance(self.dag.multi_graph.nodes[op_node_1]["op"], Gate)
        self.assertIsInstance(self.dag.multi_graph.nodes[op_node_2]["op"], Gate)

    def test_get_op_nodes_particular(self):
        self.dag.apply_operation_back(HGate(self.qubit0))
        self.dag.apply_operation_back(HGate(self.qubit1))
        self.dag.apply_operation_back(CnotGate(self.qubit0, self.qubit1))

        op_nodes = self.dag.get_op_nodes(op=HGate(self.qubit0))
        self.assertEqual(len(op_nodes), 2)

        op_node_1 = op_nodes.pop()
        op_node_2 = op_nodes.pop()
        self.assertIsInstance(self.dag.multi_graph.nodes[op_node_1]["op"], HGate)
        self.assertIsInstance(self.dag.multi_graph.nodes[op_node_2]["op"], HGate)

    def test_get_named_nodes(self):
        self.dag.apply_operation_back(CnotGate(self.qubit0, self.qubit1))
        self.dag.apply_operation_back(HGate(self.qubit0))
        self.dag.apply_operation_back(CnotGate(self.qubit2, self.qubit1))
        self.dag.apply_operation_back(CnotGate(self.qubit0, self.qubit2))
        self.dag.apply_operation_back(HGate(self.qubit2))

        # The ordering is not assured, so we only compare the output (unordered) sets.
        # We use tuples because lists aren't hashable.
        named_nodes = self.dag.get_named_nodes('cx')
        node_qargs = {tuple(self.dag.multi_graph.node[node_id]["op"].qargs)
                      for node_id in named_nodes}
        expected_qargs = {
            (self.qubit0, self.qubit1),
            (self.qubit2, self.qubit1),
            (self.qubit0, self.qubit2)}
        self.assertEqual(expected_qargs, node_qargs)

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

class TestDagLayers(QiskitTestCase):
    """Test finding layers on the dag"""

    def test_layers_basic(self):
        """ The layers() method."""
        qreg = QuantumRegister(2, 'qr')
        creg = ClassicalRegister(2, 'cr')
        qubit0 = qreg[0]
        qubit1 = qreg[1]
        clbit0 = creg[0]
        clbit1 = creg[1]
        condition = (creg, 3)
        dag = DAGCircuit()
        dag.add_basis_element('h', 1, 0, 0)
        dag.add_basis_element('cx', 2, 0, 0)
        dag.add_basis_element('x', 1, 0, 0)
        dag.add_basis_element('measure', 1, 1, 0)
        dag.add_qreg(qreg)
        dag.add_creg(creg)
        dag.apply_operation_back(HGate(qubit0))
        dag.apply_operation_back(CnotGate(qubit0, qubit1), condition=None)
        dag.apply_operation_back(Measure(qubit1, clbit1), condition=None)
        dag.apply_operation_back(XGate(qubit1), condition=condition)
        dag.apply_operation_back(Measure(qubit0, clbit0), condition=None)
        dag.apply_operation_back(Measure(qubit1, clbit1), condition=None)

        layers = list(dag.layers())
        self.assertEqual(5, len(layers))

        name_layers = [
            [node[1]["op"].name
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

<<<<<<< HEAD
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

=======

class TestDagSubstitute(QiskitTestCase):
    """Test substitutuing a dag node with a sub-dag"""
    def setUp(self):
        self.dag = DAGCircuit()        
        qreg = QuantumRegister(3, 'qr')
        creg = ClassicalRegister(2, 'cr')
        self.dag.add_qreg(qreg)
        self.dag.add_creg(creg)
        self.dag.add_basis_element(name='h', number_qubits=1,
                                   number_classical=0, number_parameters=0)
        self.dag.add_basis_element('cx', 2, 0, 0)
        self.dag.add_basis_element('x', 1, 0, 0)
        self.dag.add_basis_element('measure', 1, 1, 0)
        self.dag.add_basis_element('reset', 1, 0, 0)

        self.qubit0 = qreg[0]
        self.qubit1 = qreg[1]
        self.qubit2 = qreg[2]
        self.clbit0 = creg[0]
        self.clbit1 = creg[1]
        self.condition = (creg, 3)

        self.dag.apply_operation_back(HGate(self.qubit0))
        self.dag.apply_operation_back(CnotGate(self.qubit0, self.qubit1))
        self.dag.apply_operation_back(XGate(self.qubit1))


    def test_substitute_circuit_one_back(self):
        x_node = self.dag.get_op_nodes(op=XGate(self.qubit1)).pop()
        self.dag.substitute_circuit_one(x_node, DAGCircuit())

    def test_substitute_circuit_one_middle(self):
        pass

    def test_substitute_circuit_one_front(self):
        pass

    def test_substitute_circuit_all(self):
        pass

>>>>>>> decomposition rules in gate definitions as list of DAGs
>>>>>>> decomposition rules in gate definitions as list of DAGs

if __name__ == '__main__':
    unittest.main()
