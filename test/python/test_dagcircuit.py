# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
# pylint: disable=invalid-name

"""Test for the DAGCircuit object"""

import unittest

from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit import ClassicalRegister
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import Measure
from qiskit.circuit import Reset
from qiskit.circuit import Gate, Instruction
from qiskit.extensions.standard.h import HGate
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.x import XGate
from qiskit.extensions.standard.barrier import Barrier
from qiskit.dagcircuit.exceptions import DAGCircuitError
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase


class TestDagRegisters(QiskitTestCase):
    """Test qreg and creg inside the dag"""

    def test_add_qreg_creg(self):
        """add_qreg() and  add_creg() methods"""
        dag = DAGCircuit()
        dag.add_qreg(QuantumRegister(2, 'qr'))
        dag.add_creg(ClassicalRegister(1, 'cr'))
        self.assertDictEqual(dag.qregs, {'qr': QuantumRegister(2, 'qr')})
        self.assertDictEqual(dag.cregs, {'cr': ClassicalRegister(1, 'cr')})

    def test_dag_get_qubits(self):
        """get_qubits() method """
        dag = DAGCircuit()
        dag.add_qreg(QuantumRegister(1, 'qr1'))
        dag.add_qreg(QuantumRegister(1, 'qr10'))
        dag.add_qreg(QuantumRegister(1, 'qr0'))
        dag.add_qreg(QuantumRegister(1, 'qr3'))
        dag.add_qreg(QuantumRegister(1, 'qr4'))
        dag.add_qreg(QuantumRegister(1, 'qr6'))
        self.assertListEqual(dag.qubits(), [(QuantumRegister(1, 'qr1'), 0),
                                            (QuantumRegister(1, 'qr10'), 0),
                                            (QuantumRegister(1, 'qr0'), 0),
                                            (QuantumRegister(1, 'qr3'), 0),
                                            (QuantumRegister(1, 'qr4'), 0),
                                            (QuantumRegister(1, 'qr6'), 0)])

    def test_add_reg_duplicate(self):
        """add_qreg with the same register twice is not allowed."""
        dag = DAGCircuit()
        qr = QuantumRegister(2)
        dag.add_qreg(qr)
        self.assertRaises(DAGCircuitError, dag.add_qreg, qr)

    def test_add_reg_duplicate_name(self):
        """Adding quantum registers with the same name is not allowed."""
        dag = DAGCircuit()
        qr1 = QuantumRegister(3, 'qr')
        dag.add_qreg(qr1)
        qr2 = QuantumRegister(2, 'qr')
        self.assertRaises(DAGCircuitError, dag.add_qreg, qr2)

    def test_add_reg_bad_type(self):
        """add_qreg with a classical register is not allowed."""
        dag = DAGCircuit()
        cr = ClassicalRegister(2)
        self.assertRaises(DAGCircuitError, dag.add_qreg, cr)

    def test_rename_register(self):
        """The rename_register() method. """
        dag = DAGCircuit()
        qr = QuantumRegister(2, 'qr')
        dag.add_qreg(qr)
        dag.rename_register('qr', 'v')
        self.assertTrue('v' in dag.qregs)
        self.assertEqual(dag.qregs['v'], qr)


class TestDagOperations(QiskitTestCase):
    """Test ops inside the dag"""

    def setUp(self):
        self.dag = DAGCircuit()
        qreg = QuantumRegister(3, 'qr')
        creg = ClassicalRegister(2, 'cr')
        self.dag.add_qreg(qreg)
        self.dag.add_creg(creg)

        self.qubit0 = qreg[0]
        self.qubit1 = qreg[1]
        self.qubit2 = qreg[2]
        self.clbit0 = creg[0]
        self.clbit1 = creg[1]
        self.condition = (creg, 3)

    def test_apply_operation_back(self):
        """The apply_operation_back() method."""
        self.dag.apply_operation_back(HGate(), [self.qubit0], [], condition=None)
        self.dag.apply_operation_back(CnotGate(), [self.qubit0, self.qubit1], [], condition=None)
        self.dag.apply_operation_back(Measure(), [self.qubit1, self.clbit1], [], condition=None)
        self.dag.apply_operation_back(XGate(), [self.qubit1], [], condition=self.condition)
        self.dag.apply_operation_back(Measure(), [self.qubit0, self.clbit0], [], condition=None)
        self.dag.apply_operation_back(Measure(), [self.qubit1, self.clbit1], [], condition=None)
        self.assertEqual(len(self.dag.multi_graph.nodes), 16)
        self.assertEqual(len(self.dag.multi_graph.edges), 17)

    def test_apply_operation_front(self):
        """The apply_operation_front() method"""
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_front(Reset(), [self.qubit0], [])
        h_node = self.dag.op_nodes(op=HGate).pop()
        reset_node = self.dag.op_nodes(op=Reset).pop()

        self.assertIn(reset_node, set(self.dag.predecessors(h_node)))

    def test_get_op_nodes_all(self):
        """The method dag.op_nodes() returns all op nodes"""
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CnotGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])

        op_nodes = self.dag.op_nodes()
        self.assertEqual(len(op_nodes), 3)

        for node in op_nodes:
            self.assertIsInstance(node.op, Instruction)

    def test_get_op_nodes_particular(self):
        """The method dag.gates_nodes(op=AGate) returns all the AGate nodes"""
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(HGate(), [self.qubit1], [])
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])

        self.dag.apply_operation_back(CnotGate(), [self.qubit0, self.qubit1], [])

        op_nodes = self.dag.op_nodes(op=HGate)
        self.assertEqual(len(op_nodes), 2)

        op_node_1 = op_nodes.pop()
        op_node_2 = op_nodes.pop()

        self.assertIsInstance(op_node_1.op, HGate)
        self.assertIsInstance(op_node_2.op, HGate)

    def test_quantum_successors(self):
        """The method dag.quantum_successors() returns successors connected by quantum edges"""
        self.dag.apply_operation_back(Measure(), [self.qubit1, self.clbit1], [])
        self.dag.apply_operation_back(CnotGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])

        successor_measure = self.dag.quantum_successors(
            self.dag.named_nodes('measure').pop())
        self.assertEqual(len(successor_measure), 1)
        cnot_node = successor_measure[0]

        self.assertIsInstance(cnot_node.op, CnotGate)

        successor_cnot = self.dag.quantum_successors(cnot_node)
        self.assertEqual(len(successor_cnot), 2)
        self.assertEqual(successor_cnot[0].type, 'out')
        self.assertIsInstance(successor_cnot[1].op, Reset)

    def test_get_gates_nodes(self):
        """The method dag.gate_nodes() returns all gate nodes"""
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CnotGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])

        op_nodes = self.dag.gate_nodes()
        self.assertEqual(len(op_nodes), 2)

        op_node_1 = op_nodes.pop()
        op_node_2 = op_nodes.pop()

        self.assertIsInstance(op_node_1.op, Gate)
        self.assertIsInstance(op_node_2.op, Gate)

    def test_two_q_gates(self):
        """The method dag.twoQ_gates() returns all 2Q gate nodes"""
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CnotGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Barrier(2), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])

        op_nodes = self.dag.twoQ_gates()
        self.assertEqual(len(op_nodes), 1)

        op_node = op_nodes.pop()
        self.assertIsInstance(op_node.op, Gate)
        self.assertEqual(len(op_node.qargs), 2)

    def test_get_named_nodes(self):
        """The get_named_nodes(AName) method returns all the nodes with name AName"""
        self.dag.apply_operation_back(CnotGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CnotGate(), [self.qubit2, self.qubit1], [])
        self.dag.apply_operation_back(CnotGate(), [self.qubit0, self.qubit2], [])
        self.dag.apply_operation_back(HGate(), [self.qubit2], [])

        # The ordering is not assured, so we only compare the output (unordered) sets.
        # We use tuples because lists aren't hashable.
        named_nodes = self.dag.named_nodes('cx')

        node_qargs = {tuple(node.qargs)
                      for node in named_nodes}

        expected_qargs = {
            (self.qubit0, self.qubit1),
            (self.qubit2, self.qubit1),
            (self.qubit0, self.qubit2)}
        self.assertEqual(expected_qargs, node_qargs)

    def test_topological_nodes(self):
        """The topological_nodes() method"""
        self.dag.apply_operation_back(CnotGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CnotGate(), [self.qubit2, self.qubit1], [])
        self.dag.apply_operation_back(CnotGate(), [self.qubit0, self.qubit2], [])
        self.dag.apply_operation_back(HGate(), [self.qubit2], [])

        named_nodes = self.dag.topological_nodes()

        expected = [('qr[0]', []),
                    ('qr[1]', []),
                    ('cx', [(QuantumRegister(3, 'qr'), 0), (QuantumRegister(3, 'qr'), 1)]),
                    ('h', [(QuantumRegister(3, 'qr'), 0)]),
                    ('qr[2]', []),
                    ('cx', [(QuantumRegister(3, 'qr'), 2), (QuantumRegister(3, 'qr'), 1)]),
                    ('cx', [(QuantumRegister(3, 'qr'), 0), (QuantumRegister(3, 'qr'), 2)]),
                    ('h', [(QuantumRegister(3, 'qr'), 2)]),
                    ('qr[0]', []),
                    ('qr[1]', []),
                    ('qr[2]', []),
                    ('cr[0]', []),
                    ('cr[0]', []),
                    ('cr[1]', []),
                    ('cr[1]', [])]
        self.assertEqual(expected, [(i.name, i.qargs) for i in named_nodes])

    def test_topological_op_nodes(self):
        """The topological_op_nodes() method"""
        self.dag.apply_operation_back(CnotGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CnotGate(), [self.qubit2, self.qubit1], [])
        self.dag.apply_operation_back(CnotGate(), [self.qubit0, self.qubit2], [])
        self.dag.apply_operation_back(HGate(), [self.qubit2], [])

        named_nodes = self.dag.topological_op_nodes()

        expected = [('cx', [(QuantumRegister(3, 'qr'), 0), (QuantumRegister(3, 'qr'), 1)]),
                    ('h', [(QuantumRegister(3, 'qr'), 0)]),
                    ('cx', [(QuantumRegister(3, 'qr'), 2), (QuantumRegister(3, 'qr'), 1)]),
                    ('cx', [(QuantumRegister(3, 'qr'), 0), (QuantumRegister(3, 'qr'), 2)]),
                    ('h', [(QuantumRegister(3, 'qr'), 2)])]
        self.assertEqual(expected, [(i.name, i.qargs) for i in named_nodes])

    def test_dag_nodes_on_wire(self):
        """Test that listing the gates on a qubit/classical bit gets the correct gates"""
        self.dag.apply_operation_back(CnotGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])

        qbit = self.dag.qubits()[0]
        self.assertEqual([1, 11, 12, 2], [i._node_id for i in self.dag.nodes_on_wire(qbit)])
        self.assertEqual([11, 12],
                         [i._node_id for i in self.dag.nodes_on_wire(qbit, only_ops=True)])

        cbit = self.dag.clbits()[0]
        self.assertEqual([7, 8], [i._node_id for i in self.dag.nodes_on_wire(cbit)])
        self.assertEqual([], [i._node_id for i in self.dag.nodes_on_wire(cbit, only_ops=True)])

        (reg, _) = qbit
        with self.assertRaises(DAGCircuitError):
            next(self.dag.nodes_on_wire((reg, 7)))

    def test_dag_nodes_on_wire_multiple_successors(self):
        """
        Test that if a DAGNode has multiple successors in the DAG along one wire, they are all
        retrieved in order. This could be the case for a circuit such as

                q0_0: |0>──■─────────■──
                         ┌─┴─┐┌───┐┌─┴─┐
                q0_1: |0>┤ X ├┤ H ├┤ X ├
                         └───┘└───┘└───┘
        Both the 2nd CX gate and the H gate follow the first CX gate in the DAG, so they
        both must be returned but in the correct order.
        """
        self.dag.apply_operation_back(CnotGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit1], [])
        self.dag.apply_operation_back(CnotGate(), [self.qubit0, self.qubit1], [])

        nodes = self.dag.nodes_on_wire(self.dag.qubits()[1], only_ops=True)
        node_names = [nd.name for nd in nodes]

        self.assertEqual(node_names, ['cx', 'h', 'cx'])

    def test_remove_op_node(self):
        """Test remove_op_node method."""
        self.dag.apply_operation_back(HGate(), [self.qubit0])

        op_nodes = self.dag.gate_nodes()
        h_gate = op_nodes.pop()
        self.dag.remove_op_node(h_gate)

        self.assertEqual(len(self.dag.gate_nodes()), 0)

    def test_remove_op_node_longer(self):
        """Test remove_op_node method in a "longer" dag"""
        self.dag.apply_operation_back(CnotGate(), [self.qubit0, self.qubit1])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(CnotGate(), [self.qubit2, self.qubit1])
        self.dag.apply_operation_back(CnotGate(), [self.qubit0, self.qubit2])
        self.dag.apply_operation_back(HGate(), [self.qubit2])

        op_nodes = [node for node in self.dag.topological_op_nodes()]
        self.dag.remove_op_node(op_nodes[0])

        expected = [('h', [self.qubit0]),
                    ('cx', [self.qubit2, self.qubit1]),
                    ('cx', [self.qubit0, self.qubit2]),
                    ('h', [self.qubit2])]
        self.assertEqual(expected,
                         [(i.name, i.qargs) for i in self.dag.topological_op_nodes()])

    def test_remove_non_op_node(self):
        """Try to remove a non-op node with remove_op_node method."""
        self.dag.apply_operation_back(HGate(), [self.qubit0])

        in_node = next(self.dag.topological_nodes())
        self.assertRaises(DAGCircuitError, self.dag.remove_op_node, in_node)


class TestDagLayers(QiskitTestCase):
    """Test finding layers on the dag"""

    def test_layers_basic(self):
        """The layers() method returns a list of layers, each of them with a list of nodes."""
        qreg = QuantumRegister(2, 'qr')
        creg = ClassicalRegister(2, 'cr')
        qubit0 = qreg[0]
        qubit1 = qreg[1]
        clbit0 = creg[0]
        clbit1 = creg[1]
        condition = (creg, 3)
        dag = DAGCircuit()
        dag.add_qreg(qreg)
        dag.add_creg(creg)
        dag.apply_operation_back(HGate(), [qubit0], [])
        dag.apply_operation_back(CnotGate(), [qubit0, qubit1], [], condition=None)
        dag.apply_operation_back(Measure(), [qubit1, clbit1], [], condition=None)
        dag.apply_operation_back(XGate(), [qubit1], [], condition=condition)
        dag.apply_operation_back(Measure(), [qubit0, clbit0], [], condition=None)
        dag.apply_operation_back(Measure(), [qubit1, clbit1], [], condition=None)

        layers = list(dag.layers())
        self.assertEqual(5, len(layers))

        name_layers = [
            [node.op.name
             for node in layer["graph"].multi_graph.nodes()
             if node.type == "op"] for layer in layers]

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

        self.dag = circuit_to_dag(circ)

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


class TestCircuitSpecialCases(QiskitTestCase):
    """DAGCircuit test for special cases, usually for regression."""

    def test_circuit_depth_with_repetition(self):
        """When cx repeat, they are not "the same".
        See https://github.com/Qiskit/qiskit-terra/issues/1994
        """
        qr1 = QuantumRegister(2)
        qr2 = QuantumRegister(2)
        circ = QuantumCircuit(qr1, qr2)
        circ.h(qr1[0])
        circ.cx(qr1[1], qr2[1])
        circ.cx(qr1[1], qr2[1])
        circ.h(qr2[0])
        dag = circuit_to_dag(circ)

        self.assertEqual(dag.depth(), 2)


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
        self.dag1 = circuit_to_dag(circ1)

    def test_dag_eq(self):
        """DAG equivalence check: True."""
        circ2 = QuantumCircuit(self.qr1, self.qr2)
        circ2.cx(self.qr1[2], self.qr1[3])
        circ2.u2(0.1, 0.2, self.qr1[3])
        circ2.h(self.qr1[0])
        circ2.h(self.qr1[2])
        circ2.t(self.qr1[2])
        circ2.ch(self.qr1[2], self.qr1[1])
        circ2.ccx(self.qr2[0], self.qr2[1], self.qr1[0])
        dag2 = circuit_to_dag(circ2)

        self.assertEqual(self.dag1, dag2)

    def test_dag_neq_topology(self):
        """DAG equivalence check: False. Different topology."""
        circ2 = QuantumCircuit(self.qr1, self.qr2)
        circ2.cx(self.qr1[2], self.qr1[3])
        circ2.u2(0.1, 0.2, self.qr1[3])
        circ2.h(self.qr1[0])
        circ2.h(self.qr1[2])
        circ2.t(self.qr1[2])
        circ2.ch(self.qr1[0], self.qr1[1])  # <--- The difference: ch(qr1[2], qr1[1])
        circ2.ccx(self.qr2[0], self.qr2[1], self.qr1[0])
        dag2 = circuit_to_dag(circ2)

        self.assertNotEqual(self.dag1, dag2)

    def test_dag_neq_same_topology(self):
        """DAG equivalence check: False. Same topology."""
        circ2 = QuantumCircuit(self.qr1, self.qr2)
        circ2.cx(self.qr1[2], self.qr1[3])
        circ2.u2(0.1, 0.2, self.qr1[3])
        circ2.h(self.qr1[0])
        circ2.h(self.qr1[2])
        circ2.t(self.qr1[2])
        circ2.cx(self.qr1[2], self.qr1[1])  # <--- The difference: ch(qr1[2], qr1[1])
        circ2.ccx(self.qr2[0], self.qr2[1], self.qr1[0])
        dag2 = circuit_to_dag(circ2)

        self.assertNotEqual(self.dag1, dag2)


class TestDagSubstitute(QiskitTestCase):
    """Test substitutuing a dag node with a sub-dag"""

    def setUp(self):
        self.dag = DAGCircuit()
        qreg = QuantumRegister(3, 'qr')
        creg = ClassicalRegister(2, 'cr')
        self.dag.add_qreg(qreg)
        self.dag.add_creg(creg)

        self.qubit0 = qreg[0]
        self.qubit1 = qreg[1]
        self.qubit2 = qreg[2]
        self.clbit0 = creg[0]
        self.clbit1 = creg[1]
        self.condition = (creg, 3)

        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CnotGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(XGate(), [self.qubit1], [])

    def test_substitute_circuit_one_middle(self):
        """The method substitute_node_with_dag() replaces a in-the-middle node with a DAG."""
        cx_node = self.dag.op_nodes(op=CnotGate).pop()

        flipped_cx_circuit = DAGCircuit()
        v = QuantumRegister(2, "v")
        flipped_cx_circuit.add_qreg(v)
        flipped_cx_circuit.apply_operation_back(HGate(), [v[0]], [])
        flipped_cx_circuit.apply_operation_back(HGate(), [v[1]], [])
        flipped_cx_circuit.apply_operation_back(CnotGate(), [v[1], v[0]], [])
        flipped_cx_circuit.apply_operation_back(HGate(), [v[0]], [])
        flipped_cx_circuit.apply_operation_back(HGate(), [v[1]], [])

        self.dag.substitute_node_with_dag(cx_node, flipped_cx_circuit, wires=[v[0], v[1]])

        self.assertEqual(self.dag.count_ops()['h'], 5)

    def test_substitute_circuit_one_front(self):
        """The method substitute_node_with_dag() replaces a leaf-in-the-front node with a DAG."""
        pass

    def test_substitute_circuit_one_back(self):
        """The method substitute_node_with_dag() replaces a leaf-in-the-back node with a DAG."""
        pass


class TestDagProperties(QiskitTestCase):
    """Test the DAG properties.
    """

    def test_dag_depth_empty(self):
        """Empty circuit DAG is zero depth
        """
        q = QuantumRegister(5, 'q')
        qc = QuantumCircuit(q)
        dag = circuit_to_dag(qc)
        self.assertEqual(dag.depth(), 0)

    def test_dag_depth1(self):
        """Test DAG depth #1
        """
        q1 = QuantumRegister(3, 'q1')
        q2 = QuantumRegister(2, 'q2')
        c = ClassicalRegister(5, 'c')
        qc = QuantumCircuit(q1, q2, c)
        qc.h(q1[0])
        qc.h(q1[1])
        qc.h(q1[2])
        qc.h(q2[0])
        qc.h(q2[1])
        qc.ccx(q2[1], q1[0], q2[0])
        qc.cx(q1[0], q1[1])
        qc.cx(q1[1], q2[1])
        qc.cx(q2[1], q1[2])
        qc.cx(q1[2], q2[0])
        dag = circuit_to_dag(qc)
        self.assertEqual(dag.depth(), 6)

    def test_dag_depth2(self):
        """Test barrier increases DAG depth
        """
        q = QuantumRegister(5, 'q')
        c = ClassicalRegister(1, 'c')
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.cx(q[0], q[4])
        qc.x(q[2])
        qc.x(q[2])
        qc.x(q[2])
        qc.x(q[4])
        qc.cx(q[4], q[1])
        qc.barrier(q)
        qc.measure(q[1], c[0])
        dag = circuit_to_dag(qc)
        self.assertEqual(dag.depth(), 6)

    def test_dag_depth3(self):
        """Test DAG depth for silly circuit.
        """
        q = QuantumRegister(6, 'q')
        c = ClassicalRegister(1, 'c')
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.cx(q[0], q[1])
        qc.cx(q[1], q[2])
        qc.cx(q[2], q[3])
        qc.cx(q[3], q[4])
        qc.cx(q[4], q[5])
        qc.barrier(q[0])
        qc.barrier(q[0])
        qc.measure(q[0], c[0])
        dag = circuit_to_dag(qc)
        self.assertEqual(dag.depth(), 6)


if __name__ == '__main__':
    unittest.main()
