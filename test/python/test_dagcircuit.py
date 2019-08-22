# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""Test for the DAGCircuit object"""

import unittest

import networkx as nx

from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumRegister, Qubit
from qiskit.circuit import ClassicalRegister, Clbit
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import Measure
from qiskit.circuit import Reset
from qiskit.circuit import Gate, Instruction
from qiskit.extensions.standard.iden import IdGate
from qiskit.extensions.standard.h import HGate
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.x import XGate
from qiskit.extensions.standard.u1 import U1Gate
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
        self.assertEqual(len(list(self.dag.nodes())), 16)
        self.assertEqual(len(list(self.dag.edges())), 17)

    def test_apply_operation_back_conditional(self):
        """Test consistency of apply_operation_back with condition set."""

        # Single qubit gate conditional: qc.h(qr[2]).c_if(cr, 3)

        h_gate = HGate()
        h_gate.control = self.condition
        h_node = self.dag.apply_operation_back(
            h_gate, [self.qubit2], [], h_gate.control)

        self.assertEqual(h_node.qargs, [self.qubit2])
        self.assertEqual(h_node.cargs, [])
        self.assertEqual(h_node.condition, h_gate.control)

        self.assertEqual(
            sorted(self.dag._multi_graph.in_edges(h_node, data=True)),
            sorted([
                (self.dag.input_map[self.qubit2], h_node,
                 {'wire': Qubit(*self.qubit2), 'name': 'qr[2]'}),
                (self.dag.input_map[self.clbit0], h_node,
                 {'wire': Clbit(*self.clbit0), 'name': 'cr[0]'}),
                (self.dag.input_map[self.clbit1], h_node,
                 {'wire': Clbit(*self.clbit1), 'name': 'cr[1]'}),
            ]))

        self.assertEqual(
            sorted(self.dag._multi_graph.out_edges(h_node, data=True)),
            sorted([
                (h_node, self.dag.output_map[self.qubit2],
                 {'wire': Qubit(*self.qubit2), 'name': 'qr[2]'}),
                (h_node, self.dag.output_map[self.clbit0],
                 {'wire': Clbit(*self.clbit0), 'name': 'cr[0]'}),
                (h_node, self.dag.output_map[self.clbit1],
                 {'wire': Clbit(*self.clbit1), 'name': 'cr[1]'}),
            ]))

        self.assertTrue(nx.is_directed_acyclic_graph(self.dag._multi_graph))

    def test_apply_operation_back_conditional_measure(self):
        """Test consistency of apply_operation_back for conditional measure."""

        # Measure targeting a clbit which is not a member of the conditional
        # register. qc.measure(qr[0], cr[0]).c_if(cr2, 0)

        new_creg = ClassicalRegister(1, 'cr2')
        self.dag.add_creg(new_creg)

        meas_gate = Measure()
        meas_gate.control = (new_creg, 0)
        meas_node = self.dag.apply_operation_back(
            meas_gate, [self.qubit0], [self.clbit0], meas_gate.control)

        self.assertEqual(meas_node.qargs, [self.qubit0])
        self.assertEqual(meas_node.cargs, [self.clbit0])
        self.assertEqual(meas_node.condition, meas_gate.control)

        self.assertEqual(
            sorted(self.dag._multi_graph.in_edges(meas_node, data=True)),
            sorted([
                (self.dag.input_map[self.qubit0], meas_node,
                 {'wire': Qubit(*self.qubit0), 'name': 'qr[0]'}),
                (self.dag.input_map[self.clbit0], meas_node,
                 {'wire': Clbit(*self.clbit0), 'name': 'cr[0]'}),
                (self.dag.input_map[new_creg[0]], meas_node,
                 {'wire': Clbit(new_creg, 0), 'name': 'cr2[0]'}),
            ]))

        self.assertEqual(
            sorted(self.dag._multi_graph.out_edges(meas_node, data=True)),
            sorted([
                (meas_node, self.dag.output_map[self.qubit0],
                 {'wire': Qubit(*self.qubit0), 'name': 'qr[0]'}),
                (meas_node, self.dag.output_map[self.clbit0],
                 {'wire': Clbit(*self.clbit0), 'name': 'cr[0]'}),
                (meas_node, self.dag.output_map[new_creg[0]],
                 {'wire': Clbit(new_creg, 0), 'name': 'cr2[0]'}),
            ]))

        self.assertTrue(nx.is_directed_acyclic_graph(self.dag._multi_graph))

    def test_apply_operation_back_conditional_measure_to_self(self):
        """Test consistency of apply_operation_back for measure onto conditioning bit."""

        # Measure targeting a clbit which _is_ a member of the conditional
        # register. qc.measure(qr[0], cr[0]).c_if(cr, 3)

        meas_gate = Measure()
        meas_gate.control = self.condition
        meas_node = self.dag.apply_operation_back(
            meas_gate, [self.qubit1], [self.clbit1], self.condition)

        self.assertEqual(meas_node.qargs, [self.qubit1])
        self.assertEqual(meas_node.cargs, [self.clbit1])
        self.assertEqual(meas_node.condition, meas_gate.control)

        self.assertEqual(
            sorted(self.dag._multi_graph.in_edges(meas_node, data=True)),
            sorted([
                (self.dag.input_map[self.qubit1], meas_node,
                 {'wire': Qubit(*self.qubit1), 'name': 'qr[1]'}),
                (self.dag.input_map[self.clbit0], meas_node,
                 {'wire': Clbit(*self.clbit0), 'name': 'cr[0]'}),
                (self.dag.input_map[self.clbit1], meas_node,
                 {'wire': Clbit(*self.clbit1), 'name': 'cr[1]'}),
            ]))

        self.assertEqual(
            sorted(self.dag._multi_graph.out_edges(meas_node, data=True)),
            sorted([
                (meas_node, self.dag.output_map[self.qubit1],
                 {'wire': Qubit(*self.qubit1), 'name': 'qr[1]'}),
                (meas_node, self.dag.output_map[self.clbit0],
                 {'wire': Clbit(*self.clbit0), 'name': 'cr[0]'}),
                (meas_node, self.dag.output_map[self.clbit1],
                 {'wire': Clbit(*self.clbit1), 'name': 'cr[1]'}),
            ]))

        self.assertTrue(nx.is_directed_acyclic_graph(self.dag._multi_graph))

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
        cnot_node = next(successor_measure)
        with self.assertRaises(StopIteration):
            next(successor_measure)

        self.assertIsInstance(cnot_node.op, CnotGate)

        successor_cnot = self.dag.quantum_successors(cnot_node)
        self.assertEqual(next(successor_cnot).type, 'out')
        self.assertIsInstance(next(successor_cnot).op, Reset)
        with self.assertRaises(StopIteration):
            next(successor_cnot)

    def test_quantum_predecessors(self):
        """The method dag.quantum_predecessors() returns predecessors connected by quantum edges"""
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])
        self.dag.apply_operation_back(CnotGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Measure(), [self.qubit1, self.clbit1], [])

        predecessor_measure = self.dag.quantum_predecessors(
            self.dag.named_nodes('measure').pop())
        cnot_node = next(predecessor_measure)
        with self.assertRaises(StopIteration):
            next(predecessor_measure)

        self.assertIsInstance(cnot_node.op, CnotGate)

        predecessor_cnot = self.dag.quantum_predecessors(cnot_node)
        self.assertIsInstance(next(predecessor_cnot).op, Reset)
        self.assertEqual(next(predecessor_cnot).type, 'in')
        with self.assertRaises(StopIteration):
            next(predecessor_cnot)

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
                    ('cx', [self.qubit0, self.qubit1]),
                    ('h', [self.qubit0]),
                    ('qr[2]', []),
                    ('cx', [self.qubit2, self.qubit1]),
                    ('cx', [self.qubit0, self.qubit2]),
                    ('h', [self.qubit2]),
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

        expected = [('cx', [self.qubit0, self.qubit1]),
                    ('h', [self.qubit0]),
                    ('cx', [self.qubit2, self.qubit1]),
                    ('cx', [self.qubit0, self.qubit2]),
                    ('h', [self.qubit2])]
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

        with self.assertRaises(DAGCircuitError):
            next(self.dag.nodes_on_wire((qbit.register, 7)))

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

    def test_dag_collect_runs(self):
        """Test the collect_runs method with 3 different gates."""
        self.dag.apply_operation_back(U1Gate(3.14), [self.qubit0])
        self.dag.apply_operation_back(U1Gate(3.14), [self.qubit0])
        self.dag.apply_operation_back(U1Gate(3.14), [self.qubit0])
        self.dag.apply_operation_back(CnotGate(), [self.qubit2, self.qubit1])
        self.dag.apply_operation_back(CnotGate(), [self.qubit1, self.qubit2])
        self.dag.apply_operation_back(HGate(), [self.qubit2])
        collected_runs = self.dag.collect_runs(['u1', 'cx', 'h'])
        self.assertEqual(len(collected_runs), 3)
        for run in collected_runs:
            if run[0].name == 'cx':
                self.assertEqual(len(run), 2)
                self.assertEqual(['cx'] * 2, [x.name for x in run])
                self.assertEqual(
                    [[self.qubit2, self.qubit1], [self.qubit1, self.qubit2]],
                    [x.qargs for x in run])
            elif run[0].name == 'h':
                self.assertEqual(len(run), 1)
                self.assertEqual(['h'], [x.name for x in run])
                self.assertEqual([[self.qubit2]], [x.qargs for x in run])
            elif run[0].name == 'u1':
                self.assertEqual(len(run), 3)
                self.assertEqual(['u1'] * 3, [x.name for x in run])
                self.assertEqual(
                    [[self.qubit0], [self.qubit0], [self.qubit0]],
                    [x.qargs for x in run])
            else:
                self.fail('Unknown run encountered')

    def test_dag_collect_runs_start_with_conditional(self):
        """Test collect runs with a conditional at the start of the run."""
        self.dag.apply_operation_back(
            HGate(), [self.qubit0], condition=self.condition)
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        collected_runs = self.dag.collect_runs(['h'])
        self.assertEqual(len(collected_runs), 1)
        run = collected_runs.pop()
        self.assertEqual(len(run), 2)
        self.assertEqual(['h', 'h'], [x.name for x in run])
        self.assertEqual([[self.qubit0], [self.qubit0]],
                         [x.qargs for x in run])

    def test_dag_collect_runs_conditional_in_middle(self):
        """Test collect_runs with a conditional in the middle of a run."""
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(
            HGate(), [self.qubit0], condition=self.condition)
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        collected_runs = self.dag.collect_runs(['h'])
        # Should return 2 single h gate runs (1 before condtion, 1 after)
        self.assertEqual(len(collected_runs), 2)
        for run in collected_runs:
            self.assertEqual(len(run), 1)
            self.assertEqual(['h'], [x.name for x in run])
            self.assertEqual([[self.qubit0]], [x.qargs for x in run])


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
             for node in layer["graph"].nodes()
             if node.type == "op"] for layer in layers]

        self.assertEqual([
            ['h'],
            ['cx'],
            ['measure'],
            ['x'],
            ['measure', 'measure']
        ], name_layers)

    def test_layers_maintains_order(self):
        """Test that the layers method doesn't mess up the order of the DAG as
         reported in #2698"""
        qr = QuantumRegister(1, 'q0')

        # the order the nodes should be in
        truth = [('in', 'q0[0]', 1), ('op', 'x', 3), ('op', 'id', 4), ('out', 'q0[0]', 2)]

        # this only occurred sometimes so has to be run more than once
        # (10 times seemed to always be enough for this bug to show at least once)
        for _ in range(10):
            qc = QuantumCircuit(qr)
            qc.x(0)
            dag = circuit_to_dag(qc)
            d1 = list(dag.layers())[0]['graph']
            d1.apply_operation_back(IdGate(), [qr[0]], [])

            comp = [(nd.type, nd.name, nd._node_id) for nd in d1.topological_nodes()]
            self.assertEqual(comp, truth)


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
        """Test number of qubits + clbits in circuit."""
        self.assertEqual(self.dag.width(), 6)

    def test_circuit_num_qubits(self):
        """Test number of qubits in circuit."""
        self.assertEqual(self.dag.num_qubits(), 6)

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
    """Test substituting a dag node with a sub-dag"""

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

    def test_raise_if_substituting_dag_modifies_its_conditional(self):
        """Verify that we raise if the input dag modifies any of the bits in node.condition."""

        # Our unroller's rely on substitute_node_with_dag carrying the condition
        # from the node over to the input dag, which it does by making every
        # node in the input dag conditioned over the same bits. However, if the
        # input dag e.g. measures to one of those bits, the behavior of the
        # remainder of the DAG would be different, so detect and raise in that
        # case.

        instr = Instruction('opaque', 1, 1, [])
        instr.control = self.condition
        instr_node = self.dag.apply_operation_back(
            instr, [self.qubit0], [self.clbit1], instr.control)

        sub_dag = DAGCircuit()
        sub_qr = QuantumRegister(1, 'sqr')
        sub_cr = ClassicalRegister(1, 'scr')
        sub_dag.add_qreg(sub_qr)
        sub_dag.add_creg(sub_cr)

        sub_dag.apply_operation_back(Measure(), [sub_qr[0]], [sub_cr[0]])

        with self.assertRaises(DAGCircuitError):
            self.dag.substitute_node_with_dag(instr_node, sub_dag)


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
