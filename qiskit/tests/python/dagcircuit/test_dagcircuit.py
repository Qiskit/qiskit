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

"""Test for the DAGCircuit object"""

import unittest

from ddt import ddt, data

import retworkx as rx

from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit import ClassicalRegister, Clbit
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import Measure
from qiskit.circuit import Reset
from qiskit.circuit import Gate, Instruction
from qiskit.circuit.library.standard_gates.i import IGate
from qiskit.circuit.library.standard_gates.h import HGate
from qiskit.circuit.library.standard_gates.x import CXGate
from qiskit.circuit.library.standard_gates.z import CZGate
from qiskit.circuit.library.standard_gates.x import XGate
from qiskit.circuit.library.standard_gates.u1 import U1Gate
from qiskit.circuit.barrier import Barrier
from qiskit.dagcircuit.exceptions import DAGCircuitError
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase


def raise_if_dagcircuit_invalid(dag):
    """Validates the internal consistency of a DAGCircuit._multi_graph.
    Intended for use in testing.

    Raises:
       DAGCircuitError: if DAGCircuit._multi_graph is inconsistent.
    """

    multi_graph = dag._multi_graph

    if not rx.is_directed_acyclic_graph(multi_graph):
        raise DAGCircuitError('multi_graph is not a DAG.')

    # Every node should be of type in, out, or op.
    # All input/output nodes should be present in input_map/output_map.
    for node in dag._multi_graph.nodes():
        if node.type == 'in':
            assert node is dag.input_map[node.wire]
        elif node.type == 'out':
            assert node is dag.output_map[node.wire]
        elif node.type == 'op':
            continue
        else:
            raise DAGCircuitError('Found node of unexpected type: {}'.format(
                node.type))

    # Shape of node.op should match shape of node.
    for node in dag.op_nodes():
        assert len(node.qargs) == node.op.num_qubits
        assert len(node.cargs) == node.op.num_clbits

    # Every edge should be labled with a known wire.
    edges_outside_wires = [edge_data['wire']
                           for edge_data
                           in dag._multi_graph.edges()
                           if edge_data['wire'] not in dag.wires]
    if edges_outside_wires:
        raise DAGCircuitError('multi_graph contains one or more edges ({}) '
                              'not found in DAGCircuit.wires ({}).'.format(edges_outside_wires,
                                                                           dag.wires))

    # Every wire should have exactly one input node and one output node.
    for wire in dag.wires:
        in_node = dag.input_map[wire]
        out_node = dag.output_map[wire]

        assert in_node.wire == wire
        assert out_node.wire == wire
        assert in_node.type == 'in'
        assert out_node.type == 'out'

    # Every wire should be propagated by exactly one edge between nodes.
    for wire in dag.wires:
        cur_node_id = dag.input_map[wire]._node_id
        out_node_id = dag.output_map[wire]._node_id

        while cur_node_id != out_node_id:
            out_edges = dag._multi_graph.out_edges(cur_node_id)
            edges_to_follow = [(src, dest, data) for (src, dest, data) in out_edges
                               if data['wire'] == wire]

            assert len(edges_to_follow) == 1
            cur_node_id = edges_to_follow[0][1]

    # Wires can only terminate at input/output nodes.
    for op_node in dag.op_nodes():
        assert multi_graph.in_degree(op_node._node_id) == multi_graph.out_degree(op_node._node_id)

    # Node input/output edges should match node qarg/carg/condition.
    for node in dag.op_nodes():
        in_edges = dag._multi_graph.in_edges(node._node_id)
        out_edges = dag._multi_graph.out_edges(node._node_id)

        in_wires = {data['wire'] for src, dest, data in in_edges}
        out_wires = {data['wire'] for src, dest, data in out_edges}

        node_cond_bits = set(node.condition[0][:] if node.condition is not None else [])
        node_qubits = set(node.qargs)
        node_clbits = set(node.cargs)

        all_bits = node_qubits | node_clbits | node_cond_bits

        assert in_wires == all_bits, 'In-edge wires {} != node bits {}'.format(
            in_wires, all_bits)
        assert out_wires == all_bits, 'Out-edge wires {} != node bits {}'.format(
            out_wires, all_bits)


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
        self.assertListEqual(dag.qubits, [QuantumRegister(1, 'qr1')[0],
                                          QuantumRegister(1, 'qr10')[0],
                                          QuantumRegister(1, 'qr0')[0],
                                          QuantumRegister(1, 'qr3')[0],
                                          QuantumRegister(1, 'qr4')[0],
                                          QuantumRegister(1, 'qr6')[0]])

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


class TestDagApplyOperation(QiskitTestCase):
    """Test adding an op node to a dag."""

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
        x_gate = XGate()
        x_gate.condition = self.condition
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Measure(), [self.qubit1, self.clbit1], [])
        self.dag.apply_operation_back(x_gate, [self.qubit1], [])
        self.dag.apply_operation_back(Measure(), [self.qubit0, self.clbit0], [])
        self.dag.apply_operation_back(Measure(), [self.qubit1, self.clbit1], [])
        self.assertEqual(len(list(self.dag.nodes())), 16)
        self.assertEqual(len(list(self.dag.edges())), 17)

    def test_edges(self):
        """Test that DAGCircuit.edges() behaves as expected with ops."""
        x_gate = XGate()
        x_gate.condition = self.condition
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Measure(), [self.qubit1, self.clbit1], [])
        self.dag.apply_operation_back(x_gate, [self.qubit1], [])
        self.dag.apply_operation_back(Measure(), [self.qubit0, self.clbit0], [])
        self.dag.apply_operation_back(Measure(), [self.qubit1, self.clbit1], [])
        out_edges = self.dag.edges(self.dag.output_map.values())
        self.assertEqual(list(out_edges), [])
        in_edges = self.dag.edges(self.dag.input_map.values())
        # number of edges for input nodes should be the same as number of wires
        self.assertEqual(len(list(in_edges)), 5)

    def test_apply_operation_back_conditional(self):
        """Test consistency of apply_operation_back with condition set."""

        # Single qubit gate conditional: qc.h(qr[2]).c_if(cr, 3)

        h_gate = HGate()
        h_gate.condition = self.condition
        h_node = self.dag.apply_operation_back(
            h_gate, [self.qubit2], [])

        self.assertEqual(h_node.qargs, [self.qubit2])
        self.assertEqual(h_node.cargs, [])
        self.assertEqual(h_node.condition, h_gate.condition)

        self.assertEqual(
            sorted(self.dag._multi_graph.in_edges(h_node._node_id)),
            sorted([
                (self.dag.input_map[self.qubit2]._node_id, h_node._node_id,
                 {'wire': self.qubit2, 'name': 'qr[2]'}),
                (self.dag.input_map[self.clbit0]._node_id, h_node._node_id,
                 {'wire': self.clbit0, 'name': 'cr[0]'}),
                (self.dag.input_map[self.clbit1]._node_id, h_node._node_id,
                 {'wire': self.clbit1, 'name': 'cr[1]'}),
            ]))

        self.assertEqual(
            sorted(self.dag._multi_graph.out_edges(h_node._node_id)),
            sorted([
                (h_node._node_id, self.dag.output_map[self.qubit2]._node_id,
                 {'wire': self.qubit2, 'name': 'qr[2]'}),
                (h_node._node_id, self.dag.output_map[self.clbit0]._node_id,
                 {'wire': self.clbit0, 'name': 'cr[0]'}),
                (h_node._node_id, self.dag.output_map[self.clbit1]._node_id,
                 {'wire': self.clbit1, 'name': 'cr[1]'}),
            ]))

        self.assertTrue(rx.is_directed_acyclic_graph(self.dag._multi_graph))

    def test_apply_operation_back_conditional_measure(self):
        """Test consistency of apply_operation_back for conditional measure."""

        # Measure targeting a clbit which is not a member of the conditional
        # register. qc.measure(qr[0], cr[0]).c_if(cr2, 0)

        new_creg = ClassicalRegister(1, 'cr2')
        self.dag.add_creg(new_creg)

        meas_gate = Measure()
        meas_gate.condition = (new_creg, 0)
        meas_node = self.dag.apply_operation_back(
            meas_gate, [self.qubit0], [self.clbit0])

        self.assertEqual(meas_node.qargs, [self.qubit0])
        self.assertEqual(meas_node.cargs, [self.clbit0])
        self.assertEqual(meas_node.condition, meas_gate.condition)

        self.assertEqual(
            sorted(self.dag._multi_graph.in_edges(meas_node._node_id)),
            sorted([
                (self.dag.input_map[self.qubit0]._node_id, meas_node._node_id,
                 {'wire': self.qubit0, 'name': 'qr[0]'}),
                (self.dag.input_map[self.clbit0]._node_id, meas_node._node_id,
                 {'wire': self.clbit0, 'name': 'cr[0]'}),
                (self.dag.input_map[new_creg[0]]._node_id, meas_node._node_id,
                 {'wire': Clbit(new_creg, 0), 'name': 'cr2[0]'}),
            ]))

        self.assertEqual(
            sorted(self.dag._multi_graph.out_edges(meas_node._node_id)),
            sorted([
                (meas_node._node_id, self.dag.output_map[self.qubit0]._node_id,
                 {'wire': self.qubit0, 'name': 'qr[0]'}),
                (meas_node._node_id, self.dag.output_map[self.clbit0]._node_id,
                 {'wire': self.clbit0, 'name': 'cr[0]'}),
                (meas_node._node_id, self.dag.output_map[new_creg[0]]._node_id,
                 {'wire': Clbit(new_creg, 0), 'name': 'cr2[0]'}),
            ]))

        self.assertTrue(rx.is_directed_acyclic_graph(self.dag._multi_graph))

    def test_apply_operation_back_conditional_measure_to_self(self):
        """Test consistency of apply_operation_back for measure onto conditioning bit."""

        # Measure targeting a clbit which _is_ a member of the conditional
        # register. qc.measure(qr[0], cr[0]).c_if(cr, 3)

        meas_gate = Measure()
        meas_gate.condition = self.condition
        meas_node = self.dag.apply_operation_back(
            meas_gate, [self.qubit1], [self.clbit1])

        self.assertEqual(meas_node.qargs, [self.qubit1])
        self.assertEqual(meas_node.cargs, [self.clbit1])
        self.assertEqual(meas_node.condition, meas_gate.condition)

        self.assertEqual(
            sorted(self.dag._multi_graph.in_edges(meas_node._node_id)),
            sorted([
                (self.dag.input_map[self.qubit1]._node_id, meas_node._node_id,
                 {'wire': self.qubit1, 'name': 'qr[1]'}),
                (self.dag.input_map[self.clbit0]._node_id, meas_node._node_id,
                 {'wire': self.clbit0, 'name': 'cr[0]'}),
                (self.dag.input_map[self.clbit1]._node_id, meas_node._node_id,
                 {'wire': self.clbit1, 'name': 'cr[1]'}),
            ]))

        self.assertEqual(
            sorted(self.dag._multi_graph.out_edges(meas_node._node_id)),
            sorted([
                (meas_node._node_id, self.dag.output_map[self.qubit1]._node_id,
                 {'wire': self.qubit1, 'name': 'qr[1]'}),
                (meas_node._node_id, self.dag.output_map[self.clbit0]._node_id,
                 {'wire': self.clbit0, 'name': 'cr[0]'}),
                (meas_node._node_id, self.dag.output_map[self.clbit1]._node_id,
                 {'wire': self.clbit1, 'name': 'cr[1]'}),
            ]))

        self.assertTrue(rx.is_directed_acyclic_graph(self.dag._multi_graph))

    def test_apply_operation_front(self):
        """The apply_operation_front() method"""
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_front(Reset(), [self.qubit0], [])
        h_node = self.dag.op_nodes(op=HGate).pop()
        reset_node = self.dag.op_nodes(op=Reset).pop()

        self.assertIn(reset_node, set(self.dag.predecessors(h_node)))


class TestDagNodeSelection(QiskitTestCase):
    """Test methods that select certain dag nodes"""

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

    def test_front_layer(self):
        """The method dag.front_layer() returns first layer"""
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])

        op_nodes = self.dag.front_layer()
        self.assertEqual(len(op_nodes), 1)
        self.assertIsInstance(op_nodes[0].op, HGate)

    def test_get_op_nodes_all(self):
        """The method dag.op_nodes() returns all op nodes"""
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
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

        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])

        op_nodes = self.dag.op_nodes(op=HGate)
        self.assertEqual(len(op_nodes), 2)

        op_node_1 = op_nodes.pop()
        op_node_2 = op_nodes.pop()

        self.assertIsInstance(op_node_1.op, HGate)
        self.assertIsInstance(op_node_2.op, HGate)

    def test_quantum_successors(self):
        """The method dag.quantum_successors() returns successors connected by quantum edges"""

        # q_0: |0>─────■───|0>─
        #         ┌─┐┌─┴─┐
        # q_1: |0>┤M├┤ X ├─────
        #         └╥┘└───┘
        #  c_0: 0 ═╬═══════════
        #          ║
        #  c_1: 0 ═╩═══════════

        self.dag.apply_operation_back(Measure(), [self.qubit1, self.clbit1], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])

        successor_measure = self.dag.quantum_successors(
            self.dag.named_nodes('measure').pop())
        cnot_node = next(successor_measure)
        with self.assertRaises(StopIteration):
            next(successor_measure)

        self.assertIsInstance(cnot_node.op, CXGate)

        successor_cnot = self.dag.quantum_successors(cnot_node)
        # Ordering between Reset and out[q1] is indeterminant.

        successor1 = next(successor_cnot)
        successor2 = next(successor_cnot)
        with self.assertRaises(StopIteration):
            next(successor_cnot)

        self.assertTrue(
            (successor1.type == 'out' and isinstance(successor2.op, Reset))
            or (successor2.type == 'out' and isinstance(successor1.op, Reset))
        )

    def test_quantum_predecessors(self):
        """The method dag.quantum_predecessors() returns predecessors connected by quantum edges"""

        # q_0: |0>─|0>───■─────
        #              ┌─┴─┐┌─┐
        # q_1: |0>─────┤ X ├┤M├
        #              └───┘└╥┘
        #  c_0: 0 ═══════════╬═
        #                    ║
        #  c_1: 0 ═══════════╩═

        self.dag.apply_operation_back(Reset(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Measure(), [self.qubit1, self.clbit1], [])

        predecessor_measure = self.dag.quantum_predecessors(
            self.dag.named_nodes('measure').pop())
        cnot_node = next(predecessor_measure)
        with self.assertRaises(StopIteration):
            next(predecessor_measure)

        self.assertIsInstance(cnot_node.op, CXGate)

        predecessor_cnot = self.dag.quantum_predecessors(cnot_node)
        # Ordering between Reset and in[q1] is indeterminant.

        predecessor1 = next(predecessor_cnot)
        predecessor2 = next(predecessor_cnot)
        with self.assertRaises(StopIteration):
            next(predecessor_cnot)

        self.assertTrue(
            (predecessor1.type == 'in' and isinstance(predecessor2.op, Reset))
            or (predecessor2.type == 'in' and isinstance(predecessor1.op, Reset))
        )

    def test_get_gates_nodes(self):
        """The method dag.gate_nodes() returns all gate nodes"""
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])

        op_nodes = self.dag.gate_nodes()
        self.assertEqual(len(op_nodes), 2)

        op_node_1 = op_nodes.pop()
        op_node_2 = op_nodes.pop()

        self.assertIsInstance(op_node_1.op, Gate)
        self.assertIsInstance(op_node_2.op, Gate)

    def test_two_q_gates(self):
        """The method dag.two_qubit_ops() returns all 2Q gate operation nodes"""
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Barrier(2), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])

        op_nodes = self.dag.two_qubit_ops()
        self.assertEqual(len(op_nodes), 1)

        op_node = op_nodes.pop()
        self.assertIsInstance(op_node.op, Gate)
        self.assertEqual(len(op_node.qargs), 2)

    def test_get_named_nodes(self):
        """The get_named_nodes(AName) method returns all the nodes with name AName"""
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit2, self.qubit1], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit2], [])
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
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit2, self.qubit1], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit2], [])
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
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit2, self.qubit1], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit2], [])
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
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])

        qbit = self.dag.qubits[0]
        self.assertEqual([0, 10, 11, 1], [i._node_id for i in self.dag.nodes_on_wire(qbit)])
        self.assertEqual([10, 11],
                         [i._node_id for i in self.dag.nodes_on_wire(qbit, only_ops=True)])

        cbit = self.dag.clbits[0]
        self.assertEqual([6, 7], [i._node_id for i in self.dag.nodes_on_wire(cbit)])
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
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit1], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])

        nodes = self.dag.nodes_on_wire(self.dag.qubits[1], only_ops=True)
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
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(CXGate(), [self.qubit2, self.qubit1])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit2])
        self.dag.apply_operation_back(HGate(), [self.qubit2])

        op_nodes = list(self.dag.topological_op_nodes())
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
        self.dag.apply_operation_back(CXGate(), [self.qubit2, self.qubit1])
        self.dag.apply_operation_back(CXGate(), [self.qubit1, self.qubit2])
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
        h_gate = HGate()
        h_gate.condition = self.condition
        self.dag.apply_operation_back(
            h_gate, [self.qubit0])
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
        h_gate = HGate()
        h_gate.condition = self.condition
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(
            h_gate, [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        collected_runs = self.dag.collect_runs(['h'])
        # Should return 2 single h gate runs (1 before condition, 1 after)
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
        x_gate = XGate()
        x_gate.condition = (creg, 3)
        dag = DAGCircuit()
        dag.add_qreg(qreg)
        dag.add_creg(creg)
        dag.apply_operation_back(HGate(), [qubit0], [])
        dag.apply_operation_back(CXGate(), [qubit0, qubit1], [])
        dag.apply_operation_back(Measure(), [qubit1, clbit1], [])
        dag.apply_operation_back(x_gate, [qubit1], [])
        dag.apply_operation_back(Measure(), [qubit0, clbit0], [])
        dag.apply_operation_back(Measure(), [qubit1, clbit1], [])

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
        truth = [('in', 'q0[0]', 0), ('op', 'x', 2), ('op', 'id', 3), ('out', 'q0[0]', 1)]

        # this only occurred sometimes so has to be run more than once
        # (10 times seemed to always be enough for this bug to show at least once)
        for _ in range(10):
            qc = QuantumCircuit(qr)
            qc.x(0)
            dag = circuit_to_dag(qc)
            dag1 = list(dag.layers())[0]['graph']
            dag1.apply_operation_back(IGate(), [qr[0]], [])

            comp = [(nd.type, nd.name, nd._node_id) for nd in dag1.topological_nodes()]
            self.assertEqual(comp, truth)


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

    def test_dag_from_networkx(self):
        """Test DAG from networkx creates an expected DAGCircuit object."""
        nx_graph = self.dag1.to_networkx()
        from_nx_dag = DAGCircuit.from_networkx(nx_graph)
        self.assertEqual(self.dag1, from_nx_dag)


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
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(XGate(), [self.qubit1], [])

    def test_substitute_circuit_one_middle(self):
        """The method substitute_node_with_dag() replaces a in-the-middle node with a DAG."""
        cx_node = self.dag.op_nodes(op=CXGate).pop()

        flipped_cx_circuit = DAGCircuit()
        v = QuantumRegister(2, "v")
        flipped_cx_circuit.add_qreg(v)
        flipped_cx_circuit.apply_operation_back(HGate(), [v[0]], [])
        flipped_cx_circuit.apply_operation_back(HGate(), [v[1]], [])
        flipped_cx_circuit.apply_operation_back(CXGate(), [v[1], v[0]], [])
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
        instr.condition = self.condition
        instr_node = self.dag.apply_operation_back(
            instr, [self.qubit0], [self.clbit1])

        sub_dag = DAGCircuit()
        sub_qr = QuantumRegister(1, 'sqr')
        sub_cr = ClassicalRegister(1, 'scr')
        sub_dag.add_qreg(sub_qr)
        sub_dag.add_creg(sub_cr)

        sub_dag.apply_operation_back(Measure(), [sub_qr[0]], [sub_cr[0]])

        with self.assertRaises(DAGCircuitError):
            self.dag.substitute_node_with_dag(instr_node, sub_dag)


@ddt
class TestDagSubstituteNode(QiskitTestCase):
    """Test substituting a dagnode with a node."""

    def test_substituting_node_with_wrong_width_node_raises(self):
        """Verify replacing a node with one of a different shape raises."""
        dag = DAGCircuit()
        qr = QuantumRegister(2)
        dag.add_qreg(qr)
        node_to_be_replaced = dag.apply_operation_back(CXGate(), [qr[0], qr[1]])

        with self.assertRaises(DAGCircuitError) as _:
            dag.substitute_node(node_to_be_replaced, Measure())

    @data(True, False)
    def test_substituting_io_node_raises(self, inplace):
        """Verify replacing an io node raises."""
        dag = DAGCircuit()
        qr = QuantumRegister(1)
        dag.add_qreg(qr)

        io_node = next(dag.nodes())

        with self.assertRaises(DAGCircuitError) as _:
            dag.substitute_node(io_node, HGate(), inplace=inplace)

    @data(True, False)
    def test_substituting_node_preserves_args_condition(self, inplace):
        """Verify args and condition are preserved by a substitution."""
        dag = DAGCircuit()
        qr = QuantumRegister(2)
        cr = ClassicalRegister(1)
        dag.add_qreg(qr)
        dag.add_creg(cr)
        dag.apply_operation_back(HGate(), [qr[1]])
        cx_gate = CXGate()
        cx_gate.condition = (cr, 1)
        node_to_be_replaced = dag.apply_operation_back(cx_gate, [qr[1], qr[0]])

        dag.apply_operation_back(HGate(), [qr[1]])

        replacement_node = dag.substitute_node(node_to_be_replaced, CZGate(),
                                               inplace=inplace)

        raise_if_dagcircuit_invalid(dag)
        self.assertEqual(replacement_node.name, 'cz')
        self.assertEqual(replacement_node.qargs, [qr[1], qr[0]])
        self.assertEqual(replacement_node.cargs, [])
        self.assertEqual(replacement_node.condition, (cr, 1))

        self.assertEqual(replacement_node is node_to_be_replaced, inplace)

    @data(True, False)
    def test_substituting_node_preserves_parents_children(self, inplace):
        """Verify parents and children are preserved by a substitution."""
        qc = QuantumCircuit(3, 2)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.rz(0.1, 2)
        qc.cx(1, 2)
        qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        node_to_be_replaced = dag.named_nodes('rz')[0]
        predecessors = set(dag.predecessors(node_to_be_replaced))
        successors = set(dag.successors(node_to_be_replaced))
        ancestors = dag.ancestors(node_to_be_replaced)
        descendants = dag.descendants(node_to_be_replaced)

        replacement_node = dag.substitute_node(node_to_be_replaced, U1Gate(0.1),
                                               inplace=inplace)

        raise_if_dagcircuit_invalid(dag)
        self.assertEqual(set(dag.predecessors(replacement_node)), predecessors)
        self.assertEqual(set(dag.successors(replacement_node)), successors)
        self.assertEqual(dag.ancestors(replacement_node), ancestors)
        self.assertEqual(dag.descendants(replacement_node), descendants)

        self.assertEqual(replacement_node is node_to_be_replaced, inplace)


class TestDagProperties(QiskitTestCase):
    """Test the DAG properties.
    """

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

    def test_dag_size(self):
        """Test total number of operations in dag."""
        self.assertEqual(self.dag.size(), 7)

    def test_dag_depth(self):
        """Test dag depth."""
        self.assertEqual(self.dag.depth(), 4)

    def test_dag_width(self):
        """Test number of qubits + clbits in dag."""
        self.assertEqual(self.dag.width(), 6)

    def test_dag_num_qubits(self):
        """Test number of qubits in dag."""
        self.assertEqual(self.dag.num_qubits(), 6)

    def test_dag_operations(self):
        """Test dag operations breakdown by kind of op."""
        operations = {
            'h': 2,
            't': 1,
            'u2': 1,
            'cx': 1,
            'ch': 1,
            'ccx': 1
        }

        self.assertDictEqual(self.dag.count_ops(), operations)

    def test_dag_factors(self):
        """Test number of separable factors in circuit."""
        self.assertEqual(self.dag.num_tensor_factors(), 2)

    def test_dag_depth_empty(self):
        """Empty circuit DAG is zero depth
        """
        q = QuantumRegister(5, 'q')
        qc = QuantumCircuit(q)
        dag = circuit_to_dag(qc)
        self.assertEqual(dag.depth(), 0)

    def test_dag_idle_wires(self):
        """Test dag idle_wires."""
        wires = list(self.dag.idle_wires())
        self.assertEqual(len(wires), 0)
        wires = list(self.dag.idle_wires(['u2', 'cx']))
        self.assertEqual(len(wires), 1)

    def test_dag_depth1(self):
        """Test DAG depth #1
        """
        qr1 = QuantumRegister(3, 'q1')
        qr2 = QuantumRegister(2, 'q2')
        c = ClassicalRegister(5, 'c')
        qc = QuantumCircuit(qr1, qr2, c)
        qc.h(qr1[0])
        qc.h(qr1[1])
        qc.h(qr1[2])
        qc.h(qr2[0])
        qc.h(qr2[1])
        qc.ccx(qr2[1], qr1[0], qr2[0])
        qc.cx(qr1[0], qr1[1])
        qc.cx(qr1[1], qr2[1])
        qc.cx(qr2[1], qr1[2])
        qc.cx(qr1[2], qr2[0])
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
