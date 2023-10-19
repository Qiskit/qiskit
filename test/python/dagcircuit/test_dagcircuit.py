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

from __future__ import annotations

from collections import Counter
import unittest

from ddt import ddt, data

import rustworkx as rx
from numpy import pi

from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGInNode, DAGOutNode, DAGCircuitError
from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    Clbit,
    Qubit,
    Measure,
    Delay,
    Reset,
    Gate,
    Instruction,
    Parameter,
    Barrier,
    SwitchCaseOp,
    IfElseOp,
    WhileLoopOp,
)
from qiskit.circuit.classical import expr
from qiskit.circuit.library import IGate, HGate, CXGate, CZGate, XGate, YGate, U1Gate, RXGate
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
        raise DAGCircuitError("multi_graph is not a DAG.")

    # Every node should be of type in, out, or op.
    # All input/output nodes should be present in input_map/output_map.
    for node in dag._multi_graph.nodes():
        if isinstance(node, DAGInNode):
            assert node is dag.input_map[node.wire]
        elif isinstance(node, DAGOutNode):
            assert node is dag.output_map[node.wire]
        elif isinstance(node, DAGOpNode):
            continue
        else:
            raise DAGCircuitError(f"Found node of unexpected type: {type(node)}")

    # Shape of node.op should match shape of node.
    for node in dag.op_nodes():
        assert len(node.qargs) == node.op.num_qubits
        assert len(node.cargs) == node.op.num_clbits

    # Every edge should be labled with a known wire.
    edges_outside_wires = [
        edge_data for edge_data in dag._multi_graph.edges() if edge_data not in dag.wires
    ]
    if edges_outside_wires:
        raise DAGCircuitError(
            "multi_graph contains one or more edges ({}) "
            "not found in DAGCircuit.wires ({}).".format(edges_outside_wires, dag.wires)
        )

    # Every wire should have exactly one input node and one output node.
    for wire in dag.wires:
        in_node = dag.input_map[wire]
        out_node = dag.output_map[wire]

        assert in_node.wire == wire
        assert out_node.wire == wire
        assert isinstance(in_node, DAGInNode)
        assert isinstance(out_node, DAGOutNode)

    # Every wire should be propagated by exactly one edge between nodes.
    for wire in dag.wires:
        cur_node_id = dag.input_map[wire]._node_id
        out_node_id = dag.output_map[wire]._node_id

        while cur_node_id != out_node_id:
            out_edges = dag._multi_graph.out_edges(cur_node_id)
            edges_to_follow = [(src, dest, data) for (src, dest, data) in out_edges if data == wire]

            assert len(edges_to_follow) == 1
            cur_node_id = edges_to_follow[0][1]

    # Wires can only terminate at input/output nodes.
    op_counts = Counter()
    for op_node in dag.op_nodes():
        assert multi_graph.in_degree(op_node._node_id) == multi_graph.out_degree(op_node._node_id)
        op_counts[op_node.name] += 1
    # The _op_names attribute should match the counted op names
    assert op_counts == dag._op_names

    # Node input/output edges should match node qarg/carg/condition.
    for node in dag.op_nodes():
        in_edges = dag._multi_graph.in_edges(node._node_id)
        out_edges = dag._multi_graph.out_edges(node._node_id)

        in_wires = {data for src, dest, data in in_edges}
        out_wires = {data for src, dest, data in out_edges}

        node_cond_bits = set(
            node.op.condition[0][:] if getattr(node.op, "condition", None) is not None else []
        )
        node_qubits = set(node.qargs)
        node_clbits = set(node.cargs)

        all_bits = node_qubits | node_clbits | node_cond_bits

        assert in_wires == all_bits, f"In-edge wires {in_wires} != node bits {all_bits}"
        assert out_wires == all_bits, "Out-edge wires {} != node bits {}".format(
            out_wires, all_bits
        )


class TestDagRegisters(QiskitTestCase):
    """Test qreg and creg inside the dag"""

    def test_add_qreg_creg(self):
        """add_qreg() and  add_creg() methods"""
        dag = DAGCircuit()
        dag.add_qreg(QuantumRegister(2, "qr"))
        dag.add_creg(ClassicalRegister(1, "cr"))
        self.assertDictEqual(dag.qregs, {"qr": QuantumRegister(2, "qr")})
        self.assertDictEqual(dag.cregs, {"cr": ClassicalRegister(1, "cr")})

    def test_dag_get_qubits(self):
        """get_qubits() method"""
        dag = DAGCircuit()
        dag.add_qreg(QuantumRegister(1, "qr1"))
        dag.add_qreg(QuantumRegister(1, "qr10"))
        dag.add_qreg(QuantumRegister(1, "qr0"))
        dag.add_qreg(QuantumRegister(1, "qr3"))
        dag.add_qreg(QuantumRegister(1, "qr4"))
        dag.add_qreg(QuantumRegister(1, "qr6"))
        self.assertListEqual(
            dag.qubits,
            [
                QuantumRegister(1, "qr1")[0],
                QuantumRegister(1, "qr10")[0],
                QuantumRegister(1, "qr0")[0],
                QuantumRegister(1, "qr3")[0],
                QuantumRegister(1, "qr4")[0],
                QuantumRegister(1, "qr6")[0],
            ],
        )

    def test_add_reg_duplicate(self):
        """add_qreg with the same register twice is not allowed."""
        dag = DAGCircuit()
        qr = QuantumRegister(2)
        dag.add_qreg(qr)
        self.assertRaises(DAGCircuitError, dag.add_qreg, qr)

    def test_add_reg_duplicate_name(self):
        """Adding quantum registers with the same name is not allowed."""
        dag = DAGCircuit()
        qr1 = QuantumRegister(3, "qr")
        dag.add_qreg(qr1)
        qr2 = QuantumRegister(2, "qr")
        self.assertRaises(DAGCircuitError, dag.add_qreg, qr2)

    def test_add_reg_bad_type(self):
        """add_qreg with a classical register is not allowed."""
        dag = DAGCircuit()
        cr = ClassicalRegister(2)
        self.assertRaises(DAGCircuitError, dag.add_qreg, cr)

    def test_add_qubits_invalid_qubits(self):
        """Verify we raise if pass not a Qubit."""
        dag = DAGCircuit()

        with self.assertRaisesRegex(DAGCircuitError, "not a Qubit instance"):
            dag.add_qubits([Clbit()])

        with self.assertRaisesRegex(DAGCircuitError, "not a Qubit instance"):
            dag.add_qubits([Qubit(), Clbit(), Qubit()])

    def test_add_qubits_invalid_clbits(self):
        """Verify we raise if pass not a Clbit."""
        dag = DAGCircuit()

        with self.assertRaisesRegex(DAGCircuitError, "not a Clbit instance"):
            dag.add_clbits([Qubit()])

        with self.assertRaisesRegex(DAGCircuitError, "not a Clbit instance"):
            dag.add_clbits([Clbit(), Qubit(), Clbit()])

    def test_raise_if_bits_already_present(self):
        """Verify we raise when attempting to add a Bit already in the DAG."""
        dag = DAGCircuit()
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]

        dag.add_qubits(qubits)
        dag.add_clbits(clbits)

        with self.assertRaisesRegex(DAGCircuitError, "duplicate qubits"):
            dag.add_qubits(qubits)

        with self.assertRaisesRegex(DAGCircuitError, "duplicate clbits "):
            dag.add_clbits(clbits)

    def test_raise_if_bits_already_present_from_register(self):
        """Verify we raise when attempting to add a Bit already in the DAG."""
        dag = DAGCircuit()
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        dag.add_creg(cr)
        dag.add_qreg(qr)

        with self.assertRaisesRegex(DAGCircuitError, "duplicate qubits"):
            dag.add_qubits(qr[:])

        with self.assertRaisesRegex(DAGCircuitError, "duplicate clbits "):
            dag.add_clbits(cr[:])

    def test_adding_individual_bit(self):
        """Verify we can add a individual bits to a DAG."""
        qr = QuantumRegister(3, "qr")
        dag = DAGCircuit()
        dag.add_qreg(qr)

        new_bit = Qubit()

        dag.add_qubits([new_bit])

        self.assertEqual(dag.qubits, list(qr) + [new_bit])
        self.assertEqual(list(dag.qregs.values()), [qr])

    def test_find_bit_with_registers(self):
        """Test find_bit with a register."""
        qr = QuantumRegister(3, "qr")
        dag = DAGCircuit()
        dag.add_qreg(qr)
        res = dag.find_bit(qr[2])
        self.assertEqual(res.index, 2)
        self.assertEqual(res.registers, [(qr, 2)])

    def test_find_bit_with_registers_and_standalone(self):
        """Test find_bit with a register and standalone bit."""
        qr = QuantumRegister(3, "qr")
        dag = DAGCircuit()
        dag.add_qreg(qr)
        new_bit = Qubit()
        dag.add_qubits([new_bit])
        res = dag.find_bit(qr[2])
        self.assertEqual(res.index, 2)
        bit_res = dag.find_bit(new_bit)
        self.assertEqual(bit_res.index, 3)
        self.assertEqual(bit_res.registers, [])

    def test_find_bit_with_classical_registers_and_standalone(self):
        """Test find_bit with a register and standalone bit."""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(3, "C")
        dag = DAGCircuit()
        dag.add_qreg(qr)
        dag.add_creg(cr)
        new_bit = Qubit()
        new_clbit = Clbit()
        dag.add_qubits([new_bit])
        dag.add_clbits([new_clbit])
        res = dag.find_bit(qr[2])
        self.assertEqual(res.index, 2)
        bit_res = dag.find_bit(new_bit)
        self.assertEqual(bit_res.index, 3)
        self.assertEqual(bit_res.registers, [])
        classical_res = dag.find_bit(cr[2])
        self.assertEqual(classical_res.index, 2)
        self.assertEqual(classical_res.registers, [(cr, 2)])
        single_cl_bit_res = dag.find_bit(new_clbit)
        self.assertEqual(single_cl_bit_res.index, 3)
        self.assertEqual(single_cl_bit_res.registers, [])

    def test_find_bit_missing(self):
        """Test error when find_bit is called with missing bit."""
        qr = QuantumRegister(3, "qr")
        dag = DAGCircuit()
        dag.add_qreg(qr)
        new_bit = Qubit()
        with self.assertRaises(DAGCircuitError):
            dag.find_bit(new_bit)


class TestDagWireRemoval(QiskitTestCase):
    """Test removal of registers and idle wires."""

    def setUp(self):
        super().setUp()
        self.dag = DAGCircuit()
        self.dag.name = "Name"
        self.dag.metadata = "Metadata"
        qreg = QuantumRegister(3, "qr")
        creg0 = ClassicalRegister(2, "c0")
        creg1 = ClassicalRegister(2, "c1")
        creg2 = ClassicalRegister(name="c2", bits=list(creg1))
        clbit = Clbit()
        self.dag.add_qreg(qreg)
        self.dag.add_creg(creg0)
        self.dag.add_creg(creg1)
        self.dag.add_creg(creg2)
        self.dag.add_clbits([clbit])

        self.qreg = qreg
        self.creg0 = creg0
        self.creg1 = creg1
        self.creg1_alias = creg2
        self.individual_clbit = clbit

        self.original_cregs = [self.creg0, self.creg1, self.creg1_alias]

        # skip adding bits of creg1_alias since it's just an alias for creg1
        self.original_clbits = [
            b for reg in self.original_cregs if reg != self.creg1_alias for b in reg
        ] + [self.individual_clbit]

    def assert_cregs_equal(self, cregs, excluding=None):
        """Assert test DAG cregs match the expected values.

        Args:
            cregs (Iterable(ClassicalRegister)): the classical registers to expect
            excluding (Set(ClassicalRegister)): classical registers to remove from
            ``cregs`` before the comparison.
        """
        if excluding is None:
            excluding = set()
        self.assertEqual(
            self.dag.cregs, {creg.name: creg for creg in cregs if creg not in excluding}
        )

    def assert_clbits_equal(self, clbits, excluding=None):
        """Assert test DAG clbits match the expected values.

        Args:
            clbits (Iterable(Clbit)): the classical bits to expect
            excluding (Set(ClassicalRegister)): classical bits to remove from
            ``clbits`` before the comparison.
        """
        if excluding is None:
            excluding = set()
        self.assertEqual(self.dag.clbits, [b for b in clbits if b not in excluding])

    def test_remove_idle_creg(self):
        """Removing an idle classical register removes just the register."""
        self.dag.remove_cregs(self.creg0)

        self.assert_cregs_equal(self.original_cregs, excluding={self.creg0})
        self.assert_clbits_equal(self.original_clbits)

    def test_remove_busy_creg(self):
        """Classical registers with both busy and idle underlying bits
        can be removed, keeping underlying bits."""
        self.dag.apply_operation_back(Measure(), [self.qreg[0]], [self.creg0[0]])
        self.dag.remove_cregs(self.creg0)

        # creg removal always succeeds
        self.assert_cregs_equal(self.original_cregs, excluding={self.creg0})

        # original bits remain
        self.assert_clbits_equal(self.original_clbits)

    def test_remove_cregs_shared_bits(self):
        """Removing a classical register does not remove other classical
        registers that are simply aliases of it."""
        self.dag.remove_cregs(self.creg1)

        # Only creg1 should be deleted, not its alias
        self.assert_cregs_equal(self.original_cregs, excluding={self.creg1})
        self.assert_clbits_equal(self.original_clbits)

    def test_remove_unknown_creg(self):
        """Classical register removal of unknown registers raises."""
        unknown_creg = ClassicalRegister(1)
        with self.assertRaisesRegex(DAGCircuitError, ".*cregs not in circuit.*"):
            self.dag.remove_cregs(unknown_creg)

        self.assert_cregs_equal(self.original_cregs)
        self.assert_clbits_equal(self.original_clbits)

    def test_remove_idle_clbit(self):
        """Idle classical bits not referenced by any register can be removed."""
        self.dag.remove_clbits(self.individual_clbit)

        self.assert_cregs_equal(self.original_cregs)
        self.assert_clbits_equal(self.original_clbits, excluding={self.individual_clbit})

    def test_copy_empty_like(self):
        """Copy dag circuit metadata with copy_empty_like."""
        result_dag = self.dag.copy_empty_like()
        self.assertEqual(self.dag.name, result_dag.name)
        self.assertEqual(self.dag.metadata, result_dag.metadata)
        self.assertEqual(self.dag.clbits, result_dag.clbits)
        self.assertEqual(self.dag.qubits, result_dag.qubits)
        self.assertEqual(self.dag.cregs, result_dag.cregs)
        self.assertEqual(self.dag.qregs, result_dag.qregs)
        self.assertEqual(self.dag.duration, result_dag.duration)
        self.assertEqual(self.dag.unit, result_dag.unit)

    def test_remove_busy_clbit(self):
        """Classical bit removal of busy classical bits raises."""
        self.dag.apply_operation_back(Measure(), [self.qreg[0]], [self.individual_clbit])

        with self.assertRaisesRegex(DAGCircuitError, ".*clbits not idle.*"):
            self.dag.remove_clbits(self.individual_clbit)

        self.assert_cregs_equal(self.original_cregs)
        self.assert_clbits_equal(self.original_clbits)

    def test_remove_referenced_clbit(self):
        """Classical bit removal removes registers that reference a removed bit,
        even if they have other bits that aren't removed."""
        self.dag.remove_clbits(self.creg0[0])

        self.assert_cregs_equal(self.original_cregs, excluding={self.creg0})
        self.assert_clbits_equal(self.original_clbits, excluding={self.creg0[0]})

    def test_remove_multi_reg_referenced_clbit(self):
        """Classical bit removal removes all registers that reference a removed bit."""
        self.dag.remove_clbits(*self.creg1)

        self.assert_cregs_equal(self.original_cregs, excluding={self.creg1, self.creg1_alias})
        self.assert_clbits_equal(self.original_clbits, excluding=set(self.creg1))

    def test_remove_unknown_clbit(self):
        """Classical bit removal of unknown bits raises."""
        unknown_clbit = Clbit()
        with self.assertRaisesRegex(DAGCircuitError, ".*clbits not in circuit.*"):
            self.dag.remove_clbits(unknown_clbit)

        self.assert_cregs_equal(self.original_cregs)
        self.assert_clbits_equal(self.original_clbits)


class TestDagApplyOperation(QiskitTestCase):
    """Test adding an op node to a dag."""

    def setUp(self):
        super().setUp()
        self.dag = DAGCircuit()
        qreg = QuantumRegister(3, "qr")
        creg = ClassicalRegister(2, "cr")
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
        x_gate = XGate().c_if(*self.condition)
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Measure(), [self.qubit1], [self.clbit1])
        self.dag.apply_operation_back(x_gate, [self.qubit1], [])
        self.dag.apply_operation_back(Measure(), [self.qubit0], [self.clbit0])
        self.dag.apply_operation_back(Measure(), [self.qubit1], [self.clbit1])
        self.assertEqual(len(list(self.dag.nodes())), 16)
        self.assertEqual(len(list(self.dag.edges())), 17)

    def test_apply_operation_rejects_none(self):
        """Test that the ``apply_operation_*`` methods warn when given ``None``."""
        noop = Instruction("noop", 0, 0, [])
        with self.assertWarnsRegex(DeprecationWarning, "Passing 'None'"):
            self.dag.apply_operation_back(noop, None, ())
        with self.assertWarnsRegex(DeprecationWarning, "Passing 'None'"):
            self.dag.apply_operation_back(noop, (), None)
        with self.assertWarnsRegex(DeprecationWarning, "Passing 'None'"):
            self.dag.apply_operation_front(noop, None, ())
        with self.assertWarnsRegex(DeprecationWarning, "Passing 'None'"):
            self.dag.apply_operation_front(noop, (), None)

    def test_edges(self):
        """Test that DAGCircuit.edges() behaves as expected with ops."""
        x_gate = XGate().c_if(*self.condition)
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Measure(), [self.qubit1], [self.clbit1])
        self.dag.apply_operation_back(x_gate, [self.qubit1], [])
        self.dag.apply_operation_back(Measure(), [self.qubit0], [self.clbit0])
        self.dag.apply_operation_back(Measure(), [self.qubit1], [self.clbit1])
        out_edges = self.dag.edges(self.dag.output_map.values())
        self.assertEqual(list(out_edges), [])
        in_edges = self.dag.edges(self.dag.input_map.values())
        # number of edges for input nodes should be the same as number of wires
        self.assertEqual(len(list(in_edges)), 5)

    def test_apply_operation_back_conditional(self):
        """Test consistency of apply_operation_back with condition set."""

        # Single qubit gate conditional: qc.h(qr[2]).c_if(cr, 3)

        h_gate = HGate().c_if(*self.condition)
        h_node = self.dag.apply_operation_back(h_gate, [self.qubit2], [])

        self.assertEqual(h_node.qargs, (self.qubit2,))
        self.assertEqual(h_node.cargs, ())
        self.assertEqual(h_node.op.condition, h_gate.condition)

        self.assertEqual(
            sorted(self.dag._multi_graph.in_edges(h_node._node_id)),
            sorted(
                [
                    (self.dag.input_map[self.qubit2]._node_id, h_node._node_id, self.qubit2),
                    (self.dag.input_map[self.clbit0]._node_id, h_node._node_id, self.clbit0),
                    (self.dag.input_map[self.clbit1]._node_id, h_node._node_id, self.clbit1),
                ]
            ),
        )

        self.assertEqual(
            sorted(self.dag._multi_graph.out_edges(h_node._node_id)),
            sorted(
                [
                    (h_node._node_id, self.dag.output_map[self.qubit2]._node_id, self.qubit2),
                    (h_node._node_id, self.dag.output_map[self.clbit0]._node_id, self.clbit0),
                    (h_node._node_id, self.dag.output_map[self.clbit1]._node_id, self.clbit1),
                ]
            ),
        )

        self.assertTrue(rx.is_directed_acyclic_graph(self.dag._multi_graph))

    def test_apply_operation_back_conditional_measure(self):
        """Test consistency of apply_operation_back for conditional measure."""

        # Measure targeting a clbit which is not a member of the conditional
        # register. qc.measure(qr[0], cr[0]).c_if(cr2, 0)

        new_creg = ClassicalRegister(1, "cr2")
        self.dag.add_creg(new_creg)

        meas_gate = Measure().c_if(new_creg, 0)
        meas_node = self.dag.apply_operation_back(meas_gate, [self.qubit0], [self.clbit0])

        self.assertEqual(meas_node.qargs, (self.qubit0,))
        self.assertEqual(meas_node.cargs, (self.clbit0,))
        self.assertEqual(meas_node.op.condition, meas_gate.condition)

        self.assertEqual(
            sorted(self.dag._multi_graph.in_edges(meas_node._node_id)),
            sorted(
                [
                    (self.dag.input_map[self.qubit0]._node_id, meas_node._node_id, self.qubit0),
                    (self.dag.input_map[self.clbit0]._node_id, meas_node._node_id, self.clbit0),
                    (
                        self.dag.input_map[new_creg[0]]._node_id,
                        meas_node._node_id,
                        Clbit(new_creg, 0),
                    ),
                ]
            ),
        )

        self.assertEqual(
            sorted(self.dag._multi_graph.out_edges(meas_node._node_id)),
            sorted(
                [
                    (meas_node._node_id, self.dag.output_map[self.qubit0]._node_id, self.qubit0),
                    (meas_node._node_id, self.dag.output_map[self.clbit0]._node_id, self.clbit0),
                    (
                        meas_node._node_id,
                        self.dag.output_map[new_creg[0]]._node_id,
                        Clbit(new_creg, 0),
                    ),
                ]
            ),
        )

        self.assertTrue(rx.is_directed_acyclic_graph(self.dag._multi_graph))

    def test_apply_operation_back_conditional_measure_to_self(self):
        """Test consistency of apply_operation_back for measure onto conditioning bit."""

        # Measure targeting a clbit which _is_ a member of the conditional
        # register. qc.measure(qr[0], cr[0]).c_if(cr, 3)

        meas_gate = Measure().c_if(*self.condition)
        meas_node = self.dag.apply_operation_back(meas_gate, [self.qubit1], [self.clbit1])

        self.assertEqual(meas_node.qargs, (self.qubit1,))
        self.assertEqual(meas_node.cargs, (self.clbit1,))
        self.assertEqual(meas_node.op.condition, meas_gate.condition)

        self.assertEqual(
            sorted(self.dag._multi_graph.in_edges(meas_node._node_id)),
            sorted(
                [
                    (self.dag.input_map[self.qubit1]._node_id, meas_node._node_id, self.qubit1),
                    (self.dag.input_map[self.clbit0]._node_id, meas_node._node_id, self.clbit0),
                    (self.dag.input_map[self.clbit1]._node_id, meas_node._node_id, self.clbit1),
                ]
            ),
        )

        self.assertEqual(
            sorted(self.dag._multi_graph.out_edges(meas_node._node_id)),
            sorted(
                [
                    (meas_node._node_id, self.dag.output_map[self.qubit1]._node_id, self.qubit1),
                    (meas_node._node_id, self.dag.output_map[self.clbit0]._node_id, self.clbit0),
                    (meas_node._node_id, self.dag.output_map[self.clbit1]._node_id, self.clbit1),
                ]
            ),
        )

        self.assertTrue(rx.is_directed_acyclic_graph(self.dag._multi_graph))

    def test_apply_operation_front(self):
        """The apply_operation_front() method"""
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_front(Reset(), [self.qubit0], [])
        h_node = self.dag.op_nodes(op=HGate).pop()
        reset_node = self.dag.op_nodes(op=Reset).pop()

        self.assertIn(reset_node, set(self.dag.predecessors(h_node)))

    def test_apply_operation_expr_condition(self):
        """Test that the operation-applying functions correctly handle wires implied from `Expr`
        nodes in the `condition` field of `ControlFlowOp` instances."""
        inner = QuantumCircuit(1)
        inner.x(0)

        qr = QuantumRegister(1)
        cr1 = ClassicalRegister(2, "a")
        cr2 = ClassicalRegister(2, "b")
        clbit = Clbit()

        dag = DAGCircuit()
        dag.add_qreg(qr)
        dag.add_creg(cr1)
        dag.add_creg(cr2)
        dag.add_clbits([clbit])

        # Note that 'cr2' is not in either condition.
        expected_wires = set(qr) | set(cr1) | {clbit}

        op = IfElseOp(expr.logic_and(expr.equal(cr1, 3), expr.logic_not(clbit)), inner, None)
        node = dag.apply_operation_back(op, qr, ())
        test_wires = {wire for _source, _dest, wire in dag.edges(node)}
        self.assertIsInstance(node.op.condition, expr.Expr)
        self.assertEqual(test_wires, expected_wires)

        op = WhileLoopOp(expr.logic_or(expr.less(2, cr1), clbit), inner)
        node = dag.apply_operation_front(op, qr, ())
        test_wires = {wire for _source, _dest, wire in dag.edges(node)}
        self.assertIsInstance(node.op.condition, expr.Expr)
        self.assertEqual(test_wires, expected_wires)

    def test_apply_operation_expr_target(self):
        """Test that the operation-applying functions correctly handle wires implied from `Expr`
        nodes in the `target` field of `SwitchCaseOp`."""
        case_1 = QuantumCircuit(1)
        case_1.x(0)
        case_2 = QuantumCircuit(1)
        case_2.y(0)
        qr = QuantumRegister(1)
        cr1 = ClassicalRegister(2, "a")
        cr2 = ClassicalRegister(2, "b")
        # Note that 'cr2' is not in the condition.
        op = SwitchCaseOp(expr.bit_and(cr1, 2), [(1, case_1), (2, case_2)])

        expected_wires = set(qr) | set(cr1)

        dag = DAGCircuit()
        dag.add_qreg(qr)
        dag.add_creg(cr1)
        dag.add_creg(cr2)

        node = dag.apply_operation_back(op, qr, ())
        test_wires = {wire for _source, _dest, wire in dag.edges(node)}
        self.assertIsInstance(node.op.target, expr.Expr)
        self.assertEqual(test_wires, expected_wires)

        node = dag.apply_operation_front(op, qr, ())
        test_wires = {wire for _source, _dest, wire in dag.edges(node)}
        self.assertIsInstance(node.op.target, expr.Expr)
        self.assertEqual(test_wires, expected_wires)


class TestDagNodeSelection(QiskitTestCase):
    """Test methods that select certain dag nodes"""

    def setUp(self):
        super().setUp()
        self.dag = DAGCircuit()
        qreg = QuantumRegister(3, "qr")
        creg = ClassicalRegister(2, "cr")
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

    def test_node_representations(self):
        """Test the __repr__ methods of the DAG Nodes"""
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        # test OpNode
        cx_node = self.dag.named_nodes("cx")[0]
        self.assertEqual(
            repr(cx_node),
            f"DAGOpNode(op={cx_node.op}, qargs={cx_node.qargs}, cargs={cx_node.cargs})",
        )
        # test InNode
        predecessor_cnot = self.dag.quantum_predecessors(cx_node)
        predecessor = next(predecessor_cnot)
        self.assertEqual(repr(predecessor), f"DAGInNode(wire={predecessor.wire})")
        # test OutNode
        successor_cnot = self.dag.quantum_successors(cx_node)
        successor = next(successor_cnot)
        self.assertEqual(repr(successor), f"DAGOutNode(wire={successor.wire})")

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

        self.dag.apply_operation_back(Measure(), [self.qubit1], [self.clbit1])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])

        successor_measure = self.dag.quantum_successors(self.dag.named_nodes("measure").pop())
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
            (isinstance(successor1, DAGOutNode) and isinstance(successor2.op, Reset))
            or (isinstance(successor2, DAGOutNode) and isinstance(successor1.op, Reset))
        )

    def test_is_successor(self):
        """The method dag.is_successor(A, B) checks if node B is a successor of A"""
        self.dag.apply_operation_back(Measure(), [self.qubit1], [self.clbit1])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])

        measure_node = self.dag.named_nodes("measure")[0]
        cx_node = self.dag.named_nodes("cx")[0]
        reset_node = self.dag.named_nodes("reset")[0]

        self.assertTrue(self.dag.is_successor(measure_node, cx_node))
        self.assertFalse(self.dag.is_successor(measure_node, reset_node))
        self.assertTrue(self.dag.is_successor(cx_node, reset_node))

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
        self.dag.apply_operation_back(Measure(), [self.qubit1], [self.clbit1])

        predecessor_measure = self.dag.quantum_predecessors(self.dag.named_nodes("measure").pop())
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
            (isinstance(predecessor1, DAGInNode) and isinstance(predecessor2.op, Reset))
            or (isinstance(predecessor2, DAGInNode) and isinstance(predecessor1.op, Reset))
        )

    def test_classical_predecessors(self):
        """The method dag.classical_predecessors() returns predecessors connected by classical edges"""

        #       ┌───┐                         ┌───┐
        #  q_0: ┤ H ├──■───────────────────■──┤ H ├
        #       ├───┤┌─┴─┐               ┌─┴─┐├───┤
        #  q_1: ┤ H ├┤ X ├──■─────────■──┤ X ├┤ H ├
        #       └───┘└───┘┌─┴─┐┌───┐┌─┴─┐└───┘└───┘
        #  q_2: ──────────┤ X ├┤ H ├┤ X ├──────────
        #                 └───┘└───┘└───┘
        #  c: 5/═══════════════════════════════════

        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(HGate(), [self.qubit1], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit1, self.qubit2], [])
        self.dag.apply_operation_back(HGate(), [self.qubit2], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit1, self.qubit2], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(HGate(), [self.qubit1], [])
        self.dag.apply_operation_back(Measure(), [self.qubit0, self.clbit0], [])
        self.dag.apply_operation_back(Measure(), [self.qubit1, self.clbit1], [])

        predecessor_measure = self.dag.classical_predecessors(self.dag.named_nodes("measure").pop())

        predecessor1 = next(predecessor_measure)

        with self.assertRaises(StopIteration):
            next(predecessor_measure)

        self.assertIsInstance(predecessor1, DAGInNode)
        self.assertIsInstance(predecessor1.wire, Clbit)

    def test_classical_successors(self):
        """The method dag.classical_successors() returns successors connected by classical edges"""

        #       ┌───┐                         ┌───┐
        #  q_0: ┤ H ├──■───────────────────■──┤ H ├
        #       ├───┤┌─┴─┐               ┌─┴─┐├───┤
        #  q_1: ┤ H ├┤ X ├──■─────────■──┤ X ├┤ H ├
        #       └───┘└───┘┌─┴─┐┌───┐┌─┴─┐└───┘└───┘
        #  q_2: ──────────┤ X ├┤ H ├┤ X ├──────────
        #                 └───┘└───┘└───┘
        #  c: 5/═══════════════════════════════════

        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(HGate(), [self.qubit1], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit1, self.qubit2], [])
        self.dag.apply_operation_back(HGate(), [self.qubit2], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit1, self.qubit2], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(HGate(), [self.qubit1], [])
        self.dag.apply_operation_back(Measure(), [self.qubit0, self.clbit0], [])
        self.dag.apply_operation_back(Measure(), [self.qubit1, self.clbit1], [])

        successors_measure = self.dag.classical_successors(self.dag.named_nodes("measure").pop())

        successors1 = next(successors_measure)
        with self.assertRaises(StopIteration):
            next(successors_measure)

        self.assertIsInstance(successors1, DAGOutNode)
        self.assertIsInstance(successors1.wire, Clbit)

    def test_is_predecessor(self):
        """The method dag.is_predecessor(A, B) checks if node B is a predecessor of A"""

        self.dag.apply_operation_back(Measure(), [self.qubit1], [self.clbit1])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])

        measure_node = self.dag.named_nodes("measure")[0]
        cx_node = self.dag.named_nodes("cx")[0]
        reset_node = self.dag.named_nodes("reset")[0]

        self.assertTrue(self.dag.is_predecessor(cx_node, measure_node))
        self.assertFalse(self.dag.is_predecessor(reset_node, measure_node))
        self.assertTrue(self.dag.is_predecessor(reset_node, cx_node))

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
        named_nodes = self.dag.named_nodes("cx")

        node_qargs = {tuple(node.qargs) for node in named_nodes}

        expected_qargs = {
            (self.qubit0, self.qubit1),
            (self.qubit2, self.qubit1),
            (self.qubit0, self.qubit2),
        }
        self.assertEqual(expected_qargs, node_qargs)

    def test_topological_nodes(self):
        """The topological_nodes() method"""
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit2, self.qubit1], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit2], [])
        self.dag.apply_operation_back(HGate(), [self.qubit2], [])

        named_nodes = self.dag.topological_nodes()

        qr = self.dag.qregs["qr"]
        cr = self.dag.cregs["cr"]
        expected = [
            qr[0],
            qr[1],
            ("cx", (self.qubit0, self.qubit1)),
            ("h", (self.qubit0,)),
            qr[2],
            ("cx", (self.qubit2, self.qubit1)),
            ("cx", (self.qubit0, self.qubit2)),
            ("h", (self.qubit2,)),
            qr[0],
            qr[1],
            qr[2],
            cr[0],
            cr[0],
            cr[1],
            cr[1],
        ]
        self.assertEqual(
            [((i.op.name, i.qargs) if isinstance(i, DAGOpNode) else i.wire) for i in named_nodes],
            expected,
        )

    def test_topological_op_nodes(self):
        """The topological_op_nodes() method"""
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit2, self.qubit1], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit2], [])
        self.dag.apply_operation_back(HGate(), [self.qubit2], [])

        named_nodes = self.dag.topological_op_nodes()

        expected = [
            ("cx", (self.qubit0, self.qubit1)),
            ("h", (self.qubit0,)),
            ("cx", (self.qubit2, self.qubit1)),
            ("cx", (self.qubit0, self.qubit2)),
            ("h", (self.qubit2,)),
        ]
        self.assertEqual(expected, [(i.op.name, i.qargs) for i in named_nodes])

    def test_dag_nodes_on_wire(self):
        """Test that listing the gates on a qubit/classical bit gets the correct gates"""
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])

        qbit = self.dag.qubits[0]
        self.assertEqual([0, 10, 11, 1], [i._node_id for i in self.dag.nodes_on_wire(qbit)])
        self.assertEqual(
            [10, 11], [i._node_id for i in self.dag.nodes_on_wire(qbit, only_ops=True)]
        )

        cbit = self.dag.clbits[0]
        self.assertEqual([6, 7], [i._node_id for i in self.dag.nodes_on_wire(cbit)])
        self.assertEqual([], [i._node_id for i in self.dag.nodes_on_wire(cbit, only_ops=True)])

        with self.assertRaises(DAGCircuitError):
            next(self.dag.nodes_on_wire(QuantumRegister(5, "qr")[4]))

    def test_dag_nodes_on_wire_multiple_successors(self):
        """
        Test that if an DAGOpNode has multiple successors in the DAG along one wire, they are all
        retrieved in order. This could be the case for a circuit such as

        .. parsed-literal::

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
        node_names = [nd.op.name for nd in nodes]

        self.assertEqual(node_names, ["cx", "h", "cx"])

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

        expected = [
            ("h", (self.qubit0,)),
            ("cx", (self.qubit2, self.qubit1)),
            ("cx", (self.qubit0, self.qubit2)),
            ("h", (self.qubit2,)),
        ]
        self.assertEqual(expected, [(i.op.name, i.qargs) for i in self.dag.topological_op_nodes()])

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
        collected_runs = self.dag.collect_runs(["u1", "cx", "h"])
        self.assertEqual(len(collected_runs), 3)
        for run in collected_runs:
            if run[0].op.name == "cx":
                self.assertEqual(len(run), 2)
                self.assertEqual(["cx"] * 2, [x.op.name for x in run])
                self.assertEqual(
                    [(self.qubit2, self.qubit1), (self.qubit1, self.qubit2)], [x.qargs for x in run]
                )
            elif run[0].op.name == "h":
                self.assertEqual(len(run), 1)
                self.assertEqual(["h"], [x.op.name for x in run])
                self.assertEqual([(self.qubit2,)], [x.qargs for x in run])
            elif run[0].op.name == "u1":
                self.assertEqual(len(run), 3)
                self.assertEqual(["u1"] * 3, [x.op.name for x in run])
                self.assertEqual(
                    [(self.qubit0,), (self.qubit0,), (self.qubit0,)], [x.qargs for x in run]
                )
            else:
                self.fail("Unknown run encountered")

    def test_dag_collect_runs_start_with_conditional(self):
        """Test collect runs with a conditional at the start of the run."""
        h_gate = HGate().c_if(*self.condition)
        self.dag.apply_operation_back(h_gate, [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        collected_runs = self.dag.collect_runs(["h"])
        self.assertEqual(len(collected_runs), 1)
        run = collected_runs.pop()
        self.assertEqual(len(run), 2)
        self.assertEqual(["h", "h"], [x.op.name for x in run])
        self.assertEqual([(self.qubit0,), (self.qubit0,)], [x.qargs for x in run])

    def test_dag_collect_runs_conditional_in_middle(self):
        """Test collect_runs with a conditional in the middle of a run."""
        h_gate = HGate().c_if(*self.condition)
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(h_gate, [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        collected_runs = self.dag.collect_runs(["h"])
        # Should return 2 single h gate runs (1 before condition, 1 after)
        self.assertEqual(len(collected_runs), 2)
        for run in collected_runs:
            self.assertEqual(len(run), 1)
            self.assertEqual(["h"], [x.op.name for x in run])
            self.assertEqual([(self.qubit0,)], [x.qargs for x in run])

    def test_dag_collect_1q_runs(self):
        """Test the collect_1q_runs method with 3 different gates."""
        self.dag.apply_operation_back(Reset(), [self.qubit0])
        self.dag.apply_operation_back(Delay(100), [self.qubit0])
        self.dag.apply_operation_back(U1Gate(3.14), [self.qubit0])
        self.dag.apply_operation_back(U1Gate(3.14), [self.qubit0])
        self.dag.apply_operation_back(U1Gate(3.14), [self.qubit0])
        self.dag.apply_operation_back(CXGate(), [self.qubit2, self.qubit1])
        self.dag.apply_operation_back(CXGate(), [self.qubit1, self.qubit2])
        self.dag.apply_operation_back(HGate(), [self.qubit2])
        collected_runs = self.dag.collect_1q_runs()
        self.assertEqual(len(collected_runs), 2)
        for run in collected_runs:
            if run[0].op.name == "h":
                self.assertEqual(len(run), 1)
                self.assertEqual(["h"], [x.op.name for x in run])
                self.assertEqual([(self.qubit2,)], [x.qargs for x in run])
            elif run[0].op.name == "u1":
                self.assertEqual(len(run), 3)
                self.assertEqual(["u1"] * 3, [x.op.name for x in run])
                self.assertEqual(
                    [(self.qubit0,), (self.qubit0,), (self.qubit0,)], [x.qargs for x in run]
                )
            else:
                self.fail("Unknown run encountered")

    def test_dag_collect_1q_runs_start_with_conditional(self):
        """Test collect 1q runs with a conditional at the start of the run."""
        self.dag.apply_operation_back(Reset(), [self.qubit0])
        self.dag.apply_operation_back(Delay(100), [self.qubit0])
        h_gate = HGate().c_if(*self.condition)
        self.dag.apply_operation_back(h_gate, [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        collected_runs = self.dag.collect_1q_runs()
        self.assertEqual(len(collected_runs), 1)
        run = collected_runs.pop()
        self.assertEqual(len(run), 2)
        self.assertEqual(["h", "h"], [x.op.name for x in run])
        self.assertEqual([(self.qubit0,), (self.qubit0,)], [x.qargs for x in run])

    def test_dag_collect_1q_runs_conditional_in_middle(self):
        """Test collect_1q_runs with a conditional in the middle of a run."""
        self.dag.apply_operation_back(Reset(), [self.qubit0])
        self.dag.apply_operation_back(Delay(100), [self.qubit0])
        h_gate = HGate().c_if(*self.condition)
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(h_gate, [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        collected_runs = self.dag.collect_1q_runs()
        # Should return 2 single h gate runs (1 before condition, 1 after)
        self.assertEqual(len(collected_runs), 2)
        for run in collected_runs:
            self.assertEqual(len(run), 1)
            self.assertEqual(["h"], [x.op.name for x in run])
            self.assertEqual([(self.qubit0,)], [x.qargs for x in run])

    def test_dag_collect_1q_runs_with_parameterized_gate(self):
        """Test collect 1q splits on parameterized gates."""
        theta = Parameter("theta")
        self.dag.apply_operation_back(Reset(), [self.qubit0])
        self.dag.apply_operation_back(Delay(100), [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(U1Gate(theta), [self.qubit0])
        self.dag.apply_operation_back(XGate(), [self.qubit0])
        self.dag.apply_operation_back(XGate(), [self.qubit0])
        collected_runs = self.dag.collect_1q_runs()
        self.assertEqual(len(collected_runs), 2)
        run_gates = [[x.op.name for x in run] for run in collected_runs]
        self.assertIn(["h", "h"], run_gates)
        self.assertIn(["x", "x"], run_gates)
        self.assertNotIn("u1", [x.op.name for run in collected_runs for x in run])

    def test_dag_collect_1q_runs_with_cx_in_middle(self):
        """Test collect_1q_runs_with a cx in the middle of the run."""
        self.dag.apply_operation_back(Reset(), [self.qubit0])
        self.dag.apply_operation_back(Delay(100), [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(U1Gate(3.14), [self.qubit0])
        self.dag.apply_operation_back(U1Gate(3.14), [self.qubit1])
        self.dag.apply_operation_back(U1Gate(3.14), [self.qubit1])
        self.dag.apply_operation_back(HGate(), [self.qubit1])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1])
        self.dag.apply_operation_back(YGate(), [self.qubit0])
        self.dag.apply_operation_back(YGate(), [self.qubit0])
        self.dag.apply_operation_back(XGate(), [self.qubit1])
        self.dag.apply_operation_back(XGate(), [self.qubit1])
        collected_runs = self.dag.collect_1q_runs()
        self.assertEqual(len(collected_runs), 4)
        for run in collected_runs:
            if run[0].op.name == "h":
                self.assertEqual(len(run), 3)
                self.assertEqual(["h", "h", "u1"], [x.op.name for x in run])
                self.assertEqual([(self.qubit0,)] * 3, [x.qargs for x in run])
            elif run[0].op.name == "u1":
                self.assertEqual(len(run), 3)
                self.assertEqual(["u1", "u1", "h"], [x.op.name for x in run])
                self.assertEqual([(self.qubit1,)] * 3, [x.qargs for x in run])
            elif run[0].op.name == "x":
                self.assertEqual(len(run), 2)
                self.assertEqual(["x", "x"], [x.op.name for x in run])
                self.assertEqual([(self.qubit1,)] * 2, [x.qargs for x in run])
            elif run[0].op.name == "y":
                self.assertEqual(len(run), 2)
                self.assertEqual(["y", "y"], [x.op.name for x in run])
                self.assertEqual([(self.qubit0,)] * 2, [x.qargs for x in run])
            else:
                self.fail("Unknown run encountered")


class TestDagLayers(QiskitTestCase):
    """Test finding layers on the dag"""

    def test_layers_basic(self):
        """The layers() method returns a list of layers, each of them with a list of nodes."""
        qreg = QuantumRegister(2, "qr")
        creg = ClassicalRegister(2, "cr")
        qubit0 = qreg[0]
        qubit1 = qreg[1]
        clbit0 = creg[0]
        clbit1 = creg[1]
        x_gate = XGate().c_if(creg, 3)
        dag = DAGCircuit()
        dag.add_qreg(qreg)
        dag.add_creg(creg)
        dag.apply_operation_back(HGate(), [qubit0], [])
        dag.apply_operation_back(CXGate(), [qubit0, qubit1], [])
        dag.apply_operation_back(Measure(), [qubit1], [clbit1])
        dag.apply_operation_back(x_gate, [qubit1], [])
        dag.apply_operation_back(Measure(), [qubit0], [clbit0])
        dag.apply_operation_back(Measure(), [qubit1], [clbit1])

        layers = list(dag.layers())
        self.assertEqual(5, len(layers))

        name_layers = [
            [node.op.name for node in layer["graph"].nodes() if isinstance(node, DAGOpNode)]
            for layer in layers
        ]

        self.assertEqual([["h"], ["cx"], ["measure"], ["x"], ["measure", "measure"]], name_layers)

    def test_layers_maintains_order(self):
        """Test that the layers method doesn't mess up the order of the DAG as
        reported in #2698"""
        qr = QuantumRegister(1, "q0")

        # the order the nodes should be in
        truth = [
            (DAGInNode, qr[0], 0),
            (DAGOpNode, "x", 2),
            (DAGOpNode, "id", 3),
            (DAGOutNode, qr[0], 1),
        ]

        # this only occurred sometimes so has to be run more than once
        # (10 times seemed to always be enough for this bug to show at least once)
        for _ in range(10):
            qc = QuantumCircuit(qr)
            qc.x(0)
            dag = circuit_to_dag(qc)
            dag1 = list(dag.layers())[0]["graph"]
            dag1.apply_operation_back(IGate(), [qr[0]], [])

            comp = [
                (type(nd), nd.op.name if isinstance(nd, DAGOpNode) else nd.wire, nd._node_id)
                for nd in dag1.topological_nodes()
            ]
            self.assertEqual(comp, truth)


def _sort_key(indices: dict[Qubit, int]):
    """Return key function for sorting DAGCircuits, given a global qubit mapping."""

    def _min_active_qubit_id(dag):
        """Transform a DAGCircuit into its minimum active qubit index."""
        try:
            first_op = next(dag.topological_op_nodes())
        except StopIteration:
            return -1
        return min(indices[q] for q in first_op.qargs)

    return _min_active_qubit_id


class TestCircuitProperties(QiskitTestCase):
    """DAGCircuit properties test."""

    def setUp(self):
        #       ┌───┐                ┌───┐
        # q0_0: ┤ H ├────────────────┤ X ├──────────
        #       └───┘                └─┬─┘     ┌───┐
        # q0_1: ───────────────────────┼───────┤ H ├
        #                 ┌───┐        │  ┌───┐└─┬─┘
        # q0_2: ──■───────┤ H ├────────┼──┤ T ├──■──
        #       ┌─┴─┐┌────┴───┴─────┐  │  └───┘
        # q0_3: ┤ X ├┤ U(0,0.1,0.2) ├──┼────────────
        #       └───┘└──────────────┘  │
        # q1_0: ───────────────────────■────────────
        #                              │
        # q1_1: ───────────────────────■────────────
        super().setUp()
        qr1 = QuantumRegister(4)
        qr2 = QuantumRegister(2)
        circ = QuantumCircuit(qr1, qr2)
        circ.h(qr1[0])
        circ.cx(qr1[2], qr1[3])
        circ.h(qr1[2])
        circ.t(qr1[2])
        circ.ch(qr1[2], qr1[1])
        circ.u(0.0, 0.1, 0.2, qr1[3])
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
        operations = {"h": 2, "t": 1, "u": 1, "cx": 1, "ch": 1, "ccx": 1}

        self.assertDictEqual(self.dag.count_ops(), operations)

    def test_circuit_factors(self):
        """Test number of separable factors in circuit."""
        self.assertEqual(self.dag.num_tensor_factors(), 2)

    def test_separable_circuits(self):
        """Test separating disconnected sets of qubits in a circuit."""
        # Empty case
        dag = DAGCircuit()
        self.assertEqual(dag.separable_circuits(), [])

        # 3 disconnected qubits
        qreg = QuantumRegister(3, "q")
        creg = ClassicalRegister(2, "c")
        dag.add_qreg(qreg)
        dag.add_creg(creg)
        dag.apply_operation_back(HGate(), [qreg[0]], [])
        dag.apply_operation_back(HGate(), [qreg[1]], [])
        dag.apply_operation_back(HGate(), [qreg[2]], [])
        dag.apply_operation_back(HGate(), [qreg[2]], [])

        comp_dag1 = DAGCircuit()
        comp_dag1.add_qubits([qreg[0]])
        comp_dag1.add_creg(creg)
        comp_dag1.apply_operation_back(HGate(), [qreg[0]], [])
        comp_dag2 = DAGCircuit()
        comp_dag2.add_qubits([qreg[1]])
        comp_dag2.add_creg(creg)
        comp_dag2.apply_operation_back(HGate(), [qreg[1]], [])
        comp_dag3 = DAGCircuit()
        comp_dag3.add_qubits([qreg[2]])
        comp_dag3.add_creg(creg)
        comp_dag3.apply_operation_back(HGate(), [qreg[2]], [])
        comp_dag3.apply_operation_back(HGate(), [qreg[2]], [])

        compare_dags = [comp_dag1, comp_dag2, comp_dag3]

        # Get a mapping from qubit to qubit id in original circuit
        indices = {bit: i for i, bit in enumerate(dag.qubits)}

        # Don't rely on separable_circuits outputs to be in any order. We sort by min active qubit id
        dags = sorted(dag.separable_circuits(remove_idle_qubits=True), key=_sort_key(indices))

        self.assertEqual(dags, compare_dags)

        # 2 sets of disconnected qubits
        dag.apply_operation_back(CXGate(), [qreg[1], qreg[2]], [])

        comp_dag1 = DAGCircuit()
        comp_dag1.add_qubits([qreg[0]])
        comp_dag1.add_creg(creg)
        comp_dag1.apply_operation_back(HGate(), [qreg[0]], [])
        comp_dag2 = DAGCircuit()
        comp_dag2.add_qubits([qreg[1], qreg[2]])
        comp_dag2.add_creg(creg)
        comp_dag2.apply_operation_back(HGate(), [qreg[1]], [])
        comp_dag2.apply_operation_back(HGate(), [qreg[2]], [])
        comp_dag2.apply_operation_back(HGate(), [qreg[2]], [])
        comp_dag2.apply_operation_back(CXGate(), [qreg[1], qreg[2]], [])

        compare_dags = [comp_dag1, comp_dag2]

        # Don't rely on separable_circuits outputs to be in any order. We sort by min active qubit id
        dags = sorted(dag.separable_circuits(remove_idle_qubits=True), key=_sort_key(indices))

        self.assertEqual(dags, compare_dags)

        # One connected component
        dag.apply_operation_back(CXGate(), [qreg[0], qreg[1]], [])

        comp_dag1 = DAGCircuit()
        comp_dag1.add_qreg(qreg)
        comp_dag1.add_creg(creg)
        comp_dag1.apply_operation_back(HGate(), [qreg[0]], [])
        comp_dag1.apply_operation_back(HGate(), [qreg[1]], [])
        comp_dag1.apply_operation_back(HGate(), [qreg[2]], [])
        comp_dag1.apply_operation_back(HGate(), [qreg[2]], [])
        comp_dag1.apply_operation_back(CXGate(), [qreg[1], qreg[2]], [])
        comp_dag1.apply_operation_back(CXGate(), [qreg[0], qreg[1]], [])

        compare_dags = [comp_dag1]

        # Don't rely on separable_circuits outputs to be in any order. We sort by min active qubit id
        dags = sorted(dag.separable_circuits(remove_idle_qubits=True), key=_sort_key(indices))

        self.assertEqual(dags, compare_dags)

    def test_separable_circuits_w_measurements(self):
        """Test separating disconnected sets of qubits in a circuit with measurements."""
        # Test circuit ordering with measurements
        dag = DAGCircuit()
        qreg = QuantumRegister(3, "q")
        creg = ClassicalRegister(1, "c")
        dag.add_qreg(qreg)
        dag.add_creg(creg)
        dag.apply_operation_back(HGate(), [qreg[0]], [])
        dag.apply_operation_back(XGate(), [qreg[0]], [])
        dag.apply_operation_back(XGate(), [qreg[1]], [])
        dag.apply_operation_back(HGate(), [qreg[1]], [])
        dag.apply_operation_back(YGate(), [qreg[2]], [])
        dag.apply_operation_back(HGate(), [qreg[2]], [])
        dag.apply_operation_back(Measure(), [qreg[0]], [creg[0]])

        qc1 = QuantumCircuit(3, 1)
        qc1.h(0)
        qc1.x(0)
        qc1.measure(0, 0)
        qc2 = QuantumCircuit(3, 1)
        qc2.x(1)
        qc2.h(1)
        qc3 = QuantumCircuit(3, 1)
        qc3.y(2)
        qc3.h(2)
        qcs = [qc1, qc2, qc3]
        compare_dags = [circuit_to_dag(qc) for qc in qcs]

        # Get a mapping from qubit to qubit id in original circuit
        indices = {bit: i for i, bit in enumerate(dag.qubits)}

        # Don't rely on separable_circuits outputs to be in any order. We sort by min active qubit id
        dags = sorted(dag.separable_circuits(), key=_sort_key(indices))

        self.assertEqual(dags, compare_dags)

    def test_default_metadata_value(self):
        """Test that the default DAGCircuit metadata is valid QuantumCircuit metadata."""
        qc = QuantumCircuit(1)
        qc.metadata = self.dag.metadata
        self.assertEqual(qc.metadata, {})


class TestCircuitControlFlowProperties(QiskitTestCase):
    """Properties tests of DAGCircuit with control-flow instructions."""

    def setUp(self):
        super().setUp()
        qc = QuantumCircuit(5, 1)
        qc.h(0)
        qc.measure(0, 0)
        # The depth of an if-else is the path through the longest block (regardless of the
        # condition).  The size is the sum of both blocks (mostly for optimisation-target purposes).
        with qc.if_test((qc.clbits[0], True)) as else_:
            qc.x(1)
            qc.cx(2, 3)
        # The longest depth goes through this else_ - the block contributes 13 in size and 12 in
        # depth (because the x(1) happens concurrently with the for).
        with else_:
            qc.x(1)
            # This for loop contributes 3x to size and depth.  Its total size (and weight in the
            # depth) is 12 (as 3 * (1 + 3))
            with qc.for_loop(range(3)):
                qc.z(2)
                # This for loop contributes 3x to size and depth.
                with qc.for_loop((4, 0, 1)):
                    qc.z(2)
        # While loops contribute 1x to both size and depth, so thsi
        with qc.while_loop((qc.clbits[0], True)):
            qc.h(0)
            qc.measure(0, 0)

        self.dag = circuit_to_dag(qc)

    def test_circuit_size(self):
        """Test total number of operations in circuit."""
        # The size sums all branches of control flow, with `for` loops weighted by their number of
        # iterations (so the outer `for_loop` has a size 12 in total).  The whole `if_else` has size
        # 15, and the while-loop body counts once.
        self.assertEqual(self.dag.size(recurse=True), 19)
        with self.assertRaisesRegex(DAGCircuitError, "Size with control flow is ambiguous"):
            self.dag.size(recurse=False)

    def test_circuit_depth(self):
        """Test circuit depth."""
        # The longest depth path goes through the `h, measure`, then the `else`, then the `while`.
        # Within the `else`, the `x(1)` happens concurrently with the `for` loop.  Because each for
        # loop has 3 values, the total weight of the else is 12.
        self.assertEqual(self.dag.depth(recurse=True), 16)
        with self.assertRaisesRegex(DAGCircuitError, "Depth with control flow is ambiguous"):
            self.dag.depth(recurse=False)

    def test_circuit_width(self):
        """Test number of qubits + clbits in circuit."""
        self.assertEqual(self.dag.width(), 6)

    def test_circuit_num_qubits(self):
        """Test number of qubits in circuit."""
        self.assertEqual(self.dag.num_qubits(), 5)

    def test_circuit_operations(self):
        """Test circuit operations breakdown by kind of op."""

        self.assertDictEqual(
            self.dag.count_ops(recurse=False), {"h": 1, "measure": 1, "if_else": 1, "while_loop": 1}
        )
        self.assertDictEqual(
            self.dag.count_ops(recurse=True),
            {
                "h": 2,
                "measure": 2,
                "if_else": 1,
                "x": 2,
                "cx": 1,
                "for_loop": 2,
                "z": 2,
                "while_loop": 1,
            },
        )


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
        #        ┌───┐                ┌───┐
        # qr1_0: ┤ H ├────────────────┤ X ├──────────
        #        └───┘                └─┬─┘     ┌───┐
        # qr1_1: ───────────────────────┼───────┤ H ├
        #                  ┌───┐        │  ┌───┐└─┬─┘
        # qr1_2: ──■───────┤ H ├────────┼──┤ T ├──■──
        #        ┌─┴─┐┌────┴───┴─────┐  │  └───┘
        # qr1_3: ┤ X ├┤ U(0,0.1,0.2) ├──┼────────────
        #        └───┘└──────────────┘  │
        # qr2_0: ───────────────────────■────────────
        #                               │
        # qr2_1: ───────────────────────■────────────
        super().setUp()
        self.qr1 = QuantumRegister(4, "qr1")
        self.qr2 = QuantumRegister(2, "qr2")
        circ1 = QuantumCircuit(self.qr1, self.qr2)
        circ1.h(self.qr1[0])
        circ1.cx(self.qr1[2], self.qr1[3])
        circ1.h(self.qr1[2])
        circ1.t(self.qr1[2])
        circ1.ch(self.qr1[2], self.qr1[1])
        circ1.u(0.0, 0.1, 0.2, self.qr1[3])
        circ1.ccx(self.qr2[0], self.qr2[1], self.qr1[0])
        self.dag1 = circuit_to_dag(circ1)

    def test_dag_eq(self):
        """DAG equivalence check: True."""
        #        ┌───┐                ┌───┐
        # qr1_0: ┤ H ├────────────────┤ X ├──────────
        #        └───┘                └─┬─┘     ┌───┐
        # qr1_1: ───────────────────────┼───────┤ H ├
        #                  ┌───┐        │  ┌───┐└─┬─┘
        # qr1_2: ──■───────┤ H ├────────┼──┤ T ├──■──
        #        ┌─┴─┐┌────┴───┴─────┐  │  └───┘
        # qr1_3: ┤ X ├┤ U(0,0.1,0.2) ├──┼────────────
        #        └───┘└──────────────┘  │
        # qr2_0: ───────────────────────■────────────
        #                               │
        # qr2_1: ───────────────────────■────────────
        circ2 = QuantumCircuit(self.qr1, self.qr2)
        circ2.cx(self.qr1[2], self.qr1[3])
        circ2.u(0.0, 0.1, 0.2, self.qr1[3])
        circ2.h(self.qr1[0])
        circ2.h(self.qr1[2])
        circ2.t(self.qr1[2])
        circ2.ch(self.qr1[2], self.qr1[1])
        circ2.ccx(self.qr2[0], self.qr2[1], self.qr1[0])
        dag2 = circuit_to_dag(circ2)

        self.assertEqual(self.dag1, dag2)

    def test_dag_neq_topology(self):
        """DAG equivalence check: False. Different topology."""
        #        ┌───┐                     ┌───┐
        # qr1_0: ┤ H ├───────■─────────────┤ X ├
        #        └───┘     ┌─┴─┐           └─┬─┘
        # qr1_1: ──────────┤ H ├─────────────┼──
        #                  ├───┤      ┌───┐  │
        # qr1_2: ──■───────┤ H ├──────┤ T ├──┼──
        #        ┌─┴─┐┌────┴───┴─────┐└───┘  │
        # qr1_3: ┤ X ├┤ U(0,0.1,0.2) ├───────┼──
        #        └───┘└──────────────┘       │
        # qr2_0: ────────────────────────────■──
        #                                    │
        # qr2_1: ────────────────────────────■──
        circ2 = QuantumCircuit(self.qr1, self.qr2)
        circ2.cx(self.qr1[2], self.qr1[3])
        circ2.u(0.0, 0.1, 0.2, self.qr1[3])
        circ2.h(self.qr1[0])
        circ2.h(self.qr1[2])
        circ2.t(self.qr1[2])
        circ2.ch(self.qr1[0], self.qr1[1])  # <--- The difference: ch(qr1[2], qr1[1])
        circ2.ccx(self.qr2[0], self.qr2[1], self.qr1[0])
        dag2 = circuit_to_dag(circ2)

        self.assertNotEqual(self.dag1, dag2)

    def test_dag_neq_same_topology(self):
        """DAG equivalence check: False. Same topology."""
        #        ┌───┐                ┌───┐
        # qr1_0: ┤ H ├────────────────┤ X ├──────────
        #        └───┘                └─┬─┘     ┌───┐
        # qr1_1: ───────────────────────┼───────┤ X ├
        #                  ┌───┐        │  ┌───┐└─┬─┘
        # qr1_2: ──■───────┤ H ├────────┼──┤ T ├──■──
        #        ┌─┴─┐┌────┴───┴─────┐  │  └───┘
        # qr1_3: ┤ X ├┤ U(0,0.1,0.2) ├──┼────────────
        #        └───┘└──────────────┘  │
        # qr2_0: ───────────────────────■────────────
        #                               │
        # qr2_1: ───────────────────────■────────────
        circ2 = QuantumCircuit(self.qr1, self.qr2)
        circ2.cx(self.qr1[2], self.qr1[3])
        circ2.u(0.0, 0.1, 0.2, self.qr1[3])
        circ2.h(self.qr1[0])
        circ2.h(self.qr1[2])
        circ2.t(self.qr1[2])
        circ2.cx(self.qr1[2], self.qr1[1])  # <--- The difference: ch(qr1[2], qr1[1])
        circ2.ccx(self.qr2[0], self.qr2[1], self.qr1[0])
        dag2 = circuit_to_dag(circ2)

        self.assertNotEqual(self.dag1, dag2)

    def test_node_params_equal_unequal(self):
        """Test node params are equal or unequal."""
        qc1 = QuantumCircuit(1)
        qc2 = QuantumCircuit(1)
        qc3 = QuantumCircuit(1)
        qc1.p(pi / 4, 0)
        dag1 = circuit_to_dag(qc1)
        qc2.p(pi / 4, 0)
        dag2 = circuit_to_dag(qc2)
        qc3.p(pi / 2, 0)
        dag3 = circuit_to_dag(qc3)
        self.assertEqual(dag1, dag2)
        self.assertNotEqual(dag2, dag3)

    def test_semantic_conditions(self):
        """Test that the semantic equality is applied to the bits in conditions as well."""
        qreg = QuantumRegister(1, name="q")
        creg = ClassicalRegister(1, name="c")
        qc1 = QuantumCircuit(qreg, creg, [Clbit()])
        qc1.x(0).c_if(qc1.cregs[0], 1)
        qc1.x(0).c_if(qc1.clbits[-1], True)
        qc2 = QuantumCircuit(qreg, creg, [Clbit()])
        qc2.x(0).c_if(qc2.cregs[0], 1)
        qc2.x(0).c_if(qc2.clbits[-1], True)
        self.assertEqual(circuit_to_dag(qc1), circuit_to_dag(qc2))

        # Order of operations transposed.
        qc1 = QuantumCircuit(qreg, creg, [Clbit()])
        qc1.x(0).c_if(qc1.cregs[0], 1)
        qc1.x(0).c_if(qc1.clbits[-1], True)
        qc2 = QuantumCircuit(qreg, creg, [Clbit()])
        qc2.x(0).c_if(qc2.clbits[-1], True)
        qc2.x(0).c_if(qc2.cregs[0], 1)
        self.assertNotEqual(circuit_to_dag(qc1), circuit_to_dag(qc2))

        # Single-bit condition values not the same.
        qc1 = QuantumCircuit(qreg, creg, [Clbit()])
        qc1.x(0).c_if(qc1.cregs[0], 1)
        qc1.x(0).c_if(qc1.clbits[-1], True)
        qc2 = QuantumCircuit(qreg, creg, [Clbit()])
        qc2.x(0).c_if(qc2.cregs[0], 1)
        qc2.x(0).c_if(qc2.clbits[-1], False)
        self.assertNotEqual(circuit_to_dag(qc1), circuit_to_dag(qc2))

    def test_semantic_expr(self):
        """Test that the semantic equality is applied to the bits in `Expr` components as well."""
        cr = ClassicalRegister(3, "c1")
        clbit1 = Clbit()
        clbit2 = Clbit()

        body = QuantumCircuit(1)
        body.x(0)

        qc1 = QuantumCircuit(cr, [Qubit(), clbit1])
        qc1.if_test(expr.logic_not(clbit1), body, [0], [])
        qc2 = QuantumCircuit(cr, [Qubit(), clbit2])
        qc2.if_test(expr.logic_not(clbit2), body, [0], [])
        self.assertEqual(circuit_to_dag(qc1), circuit_to_dag(qc2))

        qc1 = QuantumCircuit(cr, [Qubit(), clbit1])
        qc2 = QuantumCircuit(cr, [Qubit(), clbit2])
        qc1.switch(expr.bit_and(cr, 5), [(1, body)], [0], [])
        qc2.switch(expr.bit_and(cr, 5), [(1, body)], [0], [])
        self.assertEqual(circuit_to_dag(qc1), circuit_to_dag(qc2))

        # Order of bits not the same.
        qc1 = QuantumCircuit(cr, [Qubit(), clbit1])
        qc1.if_test(expr.logic_not(clbit1), body, [0], [])
        qc2 = QuantumCircuit([Qubit(), clbit2], cr)
        qc2.if_test(expr.logic_not(clbit2), body, [0], [])

        qc1 = QuantumCircuit(cr, [Qubit(), clbit1])
        qc1.switch(expr.bit_and(cr, 5), [(1, body)], [0], [])
        qc2 = QuantumCircuit([Qubit(), clbit2], cr)
        qc2.switch(expr.bit_and(cr, 5), [(1, body)], [0], [])
        self.assertNotEqual(circuit_to_dag(qc1), circuit_to_dag(qc2))


class TestDagSubstitute(QiskitTestCase):
    """Test substituting a dag node with a sub-dag"""

    def setUp(self):
        super().setUp()
        self.dag = DAGCircuit()
        qreg = QuantumRegister(3, "qr")
        creg = ClassicalRegister(2, "cr")
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

        self.assertEqual(self.dag.count_ops()["h"], 5)
        expected = DAGCircuit()
        qreg = QuantumRegister(3, "qr")
        creg = ClassicalRegister(2, "cr")
        expected.add_qreg(qreg)
        expected.add_creg(creg)
        expected.apply_operation_back(HGate(), [qreg[0]], [])
        expected.apply_operation_back(HGate(), [qreg[0]], [])
        expected.apply_operation_back(HGate(), [qreg[1]], [])
        expected.apply_operation_back(CXGate(), [qreg[1], qreg[0]], [])
        expected.apply_operation_back(HGate(), [qreg[0]], [])
        expected.apply_operation_back(HGate(), [qreg[1]], [])
        expected.apply_operation_back(XGate(), [qreg[1]], [])
        self.assertEqual(self.dag, expected)

    def test_substitute_circuit_one_front(self):
        """The method substitute_node_with_dag() replaces a leaf-in-the-front node with a DAG."""
        circuit = DAGCircuit()
        v = QuantumRegister(1, "v")
        circuit.add_qreg(v)
        circuit.apply_operation_back(HGate(), [v[0]], [])
        circuit.apply_operation_back(XGate(), [v[0]], [])

        self.dag.substitute_node_with_dag(next(self.dag.topological_op_nodes()), circuit)
        expected = DAGCircuit()
        qreg = QuantumRegister(3, "qr")
        creg = ClassicalRegister(2, "cr")
        expected.add_qreg(qreg)
        expected.add_creg(creg)
        expected.apply_operation_back(HGate(), [qreg[0]], [])
        expected.apply_operation_back(XGate(), [qreg[0]], [])
        expected.apply_operation_back(CXGate(), [qreg[0], qreg[1]], [])
        expected.apply_operation_back(XGate(), [qreg[1]], [])
        self.assertEqual(self.dag, expected)

    def test_substitute_circuit_one_back(self):
        """The method substitute_node_with_dag() replaces a leaf-in-the-back node with a DAG."""
        circuit = DAGCircuit()
        v = QuantumRegister(1, "v")
        circuit.add_qreg(v)
        circuit.apply_operation_back(HGate(), [v[0]], [])
        circuit.apply_operation_back(XGate(), [v[0]], [])

        self.dag.substitute_node_with_dag(list(self.dag.topological_op_nodes())[2], circuit)
        expected = DAGCircuit()
        qreg = QuantumRegister(3, "qr")
        creg = ClassicalRegister(2, "cr")
        expected.add_qreg(qreg)
        expected.add_creg(creg)
        expected.apply_operation_back(HGate(), [qreg[0]], [])
        expected.apply_operation_back(CXGate(), [qreg[0], qreg[1]], [])
        expected.apply_operation_back(HGate(), [qreg[1]], [])
        expected.apply_operation_back(XGate(), [qreg[1]], [])

        self.assertEqual(self.dag, expected)

    def test_substitute_dag_if_else_expr(self):
        """Test that an `IfElseOp` with an `Expr` condition can be substituted for a DAG that
        contains an `IfElseOp` with an `Expr` condition."""

        body_rep = QuantumCircuit(1)
        body_rep.z(0)

        q_rep = QuantumRegister(1)
        c_rep = ClassicalRegister(2)
        replacement = DAGCircuit()
        replacement.add_qreg(q_rep)
        replacement.add_creg(c_rep)
        replacement.apply_operation_back(XGate(), [q_rep[0]], [])
        replacement.apply_operation_back(
            IfElseOp(expr.logic_not(expr.bit_and(c_rep, 1)), body_rep, None), [q_rep[0]], []
        )

        true_src = QuantumCircuit(1)
        true_src.x(0)
        true_src.z(0)
        false_src = QuantumCircuit(1)
        false_src.x(0)
        q_src = QuantumRegister(4)
        c1_src = ClassicalRegister(2)
        c2_src = ClassicalRegister(2)
        src = DAGCircuit()
        src.add_qreg(q_src)
        src.add_creg(c1_src)
        src.add_creg(c2_src)
        node = src.apply_operation_back(
            IfElseOp(expr.logic_not(expr.bit_and(c1_src, 1)), true_src, false_src), [q_src[2]], []
        )

        wires = {q_rep[0]: q_src[2], c_rep[0]: c1_src[0], c_rep[1]: c1_src[1]}
        src.substitute_node_with_dag(node, replacement, wires=wires)

        expected = DAGCircuit()
        expected.add_qreg(q_src)
        expected.add_creg(c1_src)
        expected.add_creg(c2_src)
        expected.apply_operation_back(XGate(), [q_src[2]], [])
        expected.apply_operation_back(
            IfElseOp(expr.logic_not(expr.bit_and(c1_src, 1)), body_rep, None), [q_src[2]], []
        )

        self.assertEqual(src, expected)

    def test_substitute_dag_switch_expr(self):
        """Test that a `SwitchCaseOp` with an `Expr` target can be substituted for a DAG that
        contains another `SwitchCaseOp` with an `Expr` target."""

        case_rep = QuantumCircuit(1)
        case_rep.z(0)

        q_rep = QuantumRegister(1)
        c_rep = ClassicalRegister(2)
        replacement = DAGCircuit()
        replacement.add_qreg(q_rep)
        replacement.add_creg(c_rep)
        replacement.apply_operation_back(XGate(), [q_rep[0]], [])
        # This could logically be an `IfElseOp`, but the point is just to test the `target`.
        replacement.apply_operation_back(
            SwitchCaseOp(expr.equal(c_rep[0], c_rep[1]), [(True, case_rep)]), [q_rep[0]], []
        )

        same_src = QuantumCircuit(1)
        same_src.x(0)
        same_src.z(0)
        diff_src = QuantumCircuit(1)
        diff_src.x(0)
        q_src = QuantumRegister(4)
        c1_src = ClassicalRegister(2)
        c2_src = ClassicalRegister(2)
        src = DAGCircuit()
        src.add_qreg(q_src)
        src.add_creg(c1_src)
        src.add_creg(c2_src)
        node = src.apply_operation_back(
            SwitchCaseOp(expr.lift(c1_src), [((1, 2), diff_src), ((0, 3), same_src)]),
            [q_src[2]],
            [],
        )

        wires = {q_rep[0]: q_src[2], c_rep[0]: c1_src[0], c_rep[1]: c1_src[1]}
        src.substitute_node_with_dag(node, replacement, wires=wires)

        expected = DAGCircuit()
        expected.add_qreg(q_src)
        expected.add_creg(c1_src)
        expected.add_creg(c2_src)
        expected.apply_operation_back(XGate(), [q_src[2]], [])
        node = expected.apply_operation_back(
            SwitchCaseOp(expr.equal(c1_src[0], c1_src[1]), [(True, case_rep)]), [q_src[2]], []
        )

        self.assertEqual(src, expected)

    def test_raise_if_substituting_dag_modifies_its_conditional(self):
        """Verify that we raise if the input dag modifies any of the bits in node.op.condition."""

        # The `propagate_condition=True` argument (and behaviour of `substitute_node_with_dag`
        # before the parameter was added) treats the replacement DAG as implementing only the
        # un-controlled operation.  The original contract considers it an error to replace a node
        # with an operation that may modify one of the condition bits in case this affects
        # subsequent operations, so when `propagate_condition=True`, this error behaviour is
        # maintained.

        instr = Instruction("opaque", 1, 1, [])
        instr.condition = self.condition
        instr_node = self.dag.apply_operation_back(instr, [self.qubit0], [self.clbit1])

        sub_dag = DAGCircuit()
        sub_qr = QuantumRegister(1, "sqr")
        sub_cr = ClassicalRegister(1, "scr")
        sub_dag.add_qreg(sub_qr)
        sub_dag.add_creg(sub_cr)

        sub_dag.apply_operation_back(Measure(), [sub_qr[0]], [sub_cr[0]])

        msg = "cannot propagate a condition to an element to acts on those bits"
        with self.assertRaises(DAGCircuitError, msg=msg):
            self.dag.substitute_node_with_dag(instr_node, sub_dag)

    def test_substitute_without_propagating_bit_conditional(self):
        """Test that `substitute_node_with_dag` functions when the condition is not propagated
        through to a DAG that implements the conditional logical by other means."""
        base_qubits = [Qubit(), Qubit()]
        base_clbits = [Clbit()]
        base = DAGCircuit()
        base.add_qubits(base_qubits)
        base.add_clbits(base_clbits)
        base.apply_operation_back(HGate(), [base_qubits[0]], [])
        conditioned = CZGate().to_mutable()
        conditioned.condition = (base_clbits[0], True)
        target = base.apply_operation_back(conditioned, base_qubits, [])
        base.apply_operation_back(XGate(), [base_qubits[1]], [])

        sub = QuantumCircuit(2, 1)
        sub.h(0)
        sub.cx(0, 1).c_if(0, True)
        sub.h(0)

        expected = DAGCircuit()
        expected.add_qubits(base_qubits)
        expected.add_clbits(base_clbits)
        expected.apply_operation_back(HGate(), [base_qubits[0]], [])
        expected.apply_operation_back(HGate(), [base_qubits[0]], [])
        cx = CXGate().to_mutable()
        cx.condition = (base_clbits[0], True)
        expected.apply_operation_back(cx, base_qubits, [])
        expected.apply_operation_back(HGate(), [base_qubits[0]], [])
        expected.apply_operation_back(XGate(), [base_qubits[1]], [])

        base.substitute_node_with_dag(target, circuit_to_dag(sub), propagate_condition=False)
        self.assertEqual(base, expected)

    def test_substitute_without_propagating_register_conditional(self):
        """Test that `substitute_node_with_dag` functions when the condition is not propagated
        through to a DAG that implements the conditional logical by other means."""
        base_qubits = [Qubit(), Qubit()]
        base_creg = ClassicalRegister(2)
        base = DAGCircuit()
        base.add_qubits(base_qubits)
        base.add_creg(base_creg)
        base.apply_operation_back(HGate(), [base_qubits[0]], [])
        conditioned = CZGate().to_mutable()
        conditioned.condition = (base_creg, 3)
        target = base.apply_operation_back(conditioned, base_qubits, [])
        base.apply_operation_back(XGate(), [base_qubits[1]], [])

        sub = QuantumCircuit(QuantumRegister(2), ClassicalRegister(2))
        sub.h(0)
        sub.cx(0, 1).c_if(sub.cregs[0], 3)
        sub.h(0)

        expected = DAGCircuit()
        expected.add_qubits(base_qubits)
        expected.add_creg(base_creg)
        expected.apply_operation_back(HGate(), [base_qubits[0]], [])
        expected.apply_operation_back(HGate(), [base_qubits[0]], [])
        cx = CXGate().to_mutable()
        cx.condition = (base_creg, 3)
        expected.apply_operation_back(cx, base_qubits, [])
        expected.apply_operation_back(HGate(), [base_qubits[0]], [])
        expected.apply_operation_back(XGate(), [base_qubits[1]], [])

        base.substitute_node_with_dag(target, circuit_to_dag(sub), propagate_condition=False)
        self.assertEqual(base, expected)

    def test_substitute_with_provided_wire_map_propagate_condition(self):
        """Test that the ``wires`` argument can be a direct mapping while propagating the
        condition."""
        base_qubits = [Qubit(), Qubit()]
        base_clbits = [Clbit()]
        base = DAGCircuit()
        base.add_qubits(base_qubits)
        base.add_clbits(base_clbits)
        base.apply_operation_back(HGate(), [base_qubits[0]], [])
        conditioned_cz = CZGate().to_mutable()
        conditioned_cz.condition = (base_clbits[0], True)
        target = base.apply_operation_back(conditioned_cz, base_qubits, [])
        base.apply_operation_back(XGate(), [base_qubits[1]], [])

        # Note that `sub` here doesn't have any classical bits at all.
        sub = QuantumCircuit(2)
        sub.h(0)
        sub.cx(0, 1)
        sub.h(0)

        conditioned_h = HGate().c_if(*conditioned_cz.condition)
        conditioned_cx = CXGate().c_if(*conditioned_cz.condition)

        expected = DAGCircuit()
        expected.add_qubits(base_qubits)
        expected.add_clbits(base_clbits)
        expected.apply_operation_back(HGate(), [base_qubits[0]], [])
        expected.apply_operation_back(conditioned_h.copy(), [base_qubits[0]], [])
        expected.apply_operation_back(conditioned_cx, base_qubits, [])
        expected.apply_operation_back(conditioned_h.copy(), [base_qubits[0]], [])
        expected.apply_operation_back(XGate(), [base_qubits[1]], [])

        wire_map = dict(zip(sub.qubits, base_qubits))
        base.substitute_node_with_dag(
            target, circuit_to_dag(sub), propagate_condition=True, wires=wire_map
        )
        self.assertEqual(base, expected)

    def test_substitute_with_provided_wire_map_no_propagate_condition(self):
        """Test that the ``wires`` argument can be a direct mapping while not propagating the
        condition."""
        base_qubits = [Qubit(), Qubit()]
        base_clbits = [Clbit()]
        base = DAGCircuit()
        base.add_qubits(base_qubits)
        base.add_clbits(base_clbits)
        base.apply_operation_back(HGate(), [base_qubits[0]], [])
        conditioned_cz = CZGate().to_mutable()
        conditioned_cz.condition = (base_clbits[0], True)
        target = base.apply_operation_back(conditioned_cz, base_qubits, [])
        base.apply_operation_back(XGate(), [base_qubits[1]], [])

        sub = QuantumCircuit(2, 1)
        sub.h(0)
        sub.cx(0, 1).c_if(0, True)
        sub.h(0)

        conditioned_cx = CXGate().to_mutable()
        conditioned_cx.condition = conditioned_cz.condition

        expected = DAGCircuit()
        expected.add_qubits(base_qubits)
        expected.add_clbits(base_clbits)
        expected.apply_operation_back(HGate(), [base_qubits[0]], [])
        expected.apply_operation_back(HGate(), [base_qubits[0]], [])
        expected.apply_operation_back(conditioned_cx, base_qubits, [])
        expected.apply_operation_back(HGate(), [base_qubits[0]], [])
        expected.apply_operation_back(XGate(), [base_qubits[1]], [])

        wire_map = dict(zip(sub.qubits, base_qubits))
        wire_map[sub.clbits[0]] = base_clbits[0]
        base.substitute_node_with_dag(
            target, circuit_to_dag(sub), propagate_condition=False, wires=wire_map
        )
        self.assertEqual(base, expected)

    def test_creates_additional_alias_register(self):
        """Test that a sub-condition that has no associated register in the main DAG will be
        translated faithfully by adding a new aliasing register."""
        base_qreg = QuantumRegister(2)
        base_creg = ClassicalRegister(3)
        base = DAGCircuit()
        base.add_qreg(base_qreg)
        base.add_creg(base_creg)
        target = base.apply_operation_back(Instruction("dummy", 2, 2, []), base_qreg, base_creg[:2])

        sub = QuantumCircuit(QuantumRegister(2), ClassicalRegister(2))
        sub.cx(0, 1).c_if(sub.cregs[0], 3)

        base.substitute_node_with_dag(target, circuit_to_dag(sub))

        # Check that a new register was added, and retrieve it to verify the bits in it.
        self.assertEqual(len(base.cregs), 2)
        cregs = base.cregs.copy()
        cregs.pop(base_creg.name)
        added_creg = tuple(cregs.values())[0]
        self.assertEqual(list(added_creg), list(base_creg[:2]))

        expected = DAGCircuit()
        expected.add_qreg(base_qreg)
        expected.add_creg(base_creg)
        expected.add_creg(added_creg)
        cx = CXGate().to_mutable()
        cx.condition = (added_creg, 3)
        expected.apply_operation_back(cx, base_qreg, [])
        self.assertEqual(base, expected)


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
        cx_gate = CXGate().to_mutable()
        cx_gate.condition = (cr, 1)
        node_to_be_replaced = dag.apply_operation_back(cx_gate, [qr[1], qr[0]])

        dag.apply_operation_back(HGate(), [qr[1]])

        replacement_node = dag.substitute_node(node_to_be_replaced, CZGate(), inplace=inplace)

        raise_if_dagcircuit_invalid(dag)
        self.assertEqual(replacement_node.op.name, "cz")
        self.assertEqual(replacement_node.qargs, (qr[1], qr[0]))
        self.assertEqual(replacement_node.cargs, ())
        self.assertEqual(replacement_node.op.condition, (cr, 1))

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
        node_to_be_replaced = dag.named_nodes("rz")[0]
        predecessors = set(dag.predecessors(node_to_be_replaced))
        successors = set(dag.successors(node_to_be_replaced))
        ancestors = dag.ancestors(node_to_be_replaced)
        descendants = dag.descendants(node_to_be_replaced)

        replacement_node = dag.substitute_node(node_to_be_replaced, U1Gate(0.1), inplace=inplace)

        raise_if_dagcircuit_invalid(dag)
        self.assertEqual(set(dag.predecessors(replacement_node)), predecessors)
        self.assertEqual(set(dag.successors(replacement_node)), successors)
        self.assertEqual(dag.ancestors(replacement_node), ancestors)
        self.assertEqual(dag.descendants(replacement_node), descendants)
        self.assertEqual(replacement_node is node_to_be_replaced, inplace)

    @data(True, False)
    def test_refuses_to_overwrite_condition(self, inplace):
        """Test that the method will not forcibly overwrite a condition."""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(2)

        dag = DAGCircuit()
        dag.add_qreg(qr)
        dag.add_creg(cr)
        node = dag.apply_operation_back(XGate().c_if(cr, 2), qr, [])

        with self.assertRaisesRegex(DAGCircuitError, "Cannot propagate a condition"):
            dag.substitute_node(
                node, XGate().c_if(cr, 1), inplace=inplace, propagate_condition=True
            )

    @data(True, False)
    def test_replace_if_else_op_with_another(self, inplace):
        """Test that one `IfElseOp` can be replaced with another."""
        body = QuantumCircuit(1)
        body.x(0)

        qr = QuantumRegister(1)
        cr1 = ClassicalRegister(2)
        cr2 = ClassicalRegister(2)
        dag = DAGCircuit()
        dag.add_qreg(qr)
        dag.add_creg(cr1)
        dag.add_creg(cr2)
        node = dag.apply_operation_back(IfElseOp(expr.logic_not(cr1), body.copy(), None), qr, [])
        dag.substitute_node(node, IfElseOp(expr.equal(cr1, 0), body.copy(), None), inplace=inplace)

        expected = DAGCircuit()
        expected.add_qreg(qr)
        expected.add_creg(cr1)
        expected.add_creg(cr2)
        expected.apply_operation_back(IfElseOp(expr.equal(cr1, 0), body.copy(), None), qr, [])

        self.assertEqual(dag, expected)

    @data(True, False)
    def test_reject_replace_if_else_op_with_other_resources(self, inplace):
        """Test that the resources in the `condition` of a `IfElseOp` are checked against those in
        the node to be replaced."""
        body = QuantumCircuit(1)
        body.x(0)

        qr = QuantumRegister(1)
        cr1 = ClassicalRegister(2)
        cr2 = ClassicalRegister(2)
        dag = DAGCircuit()
        dag.add_qreg(qr)
        dag.add_creg(cr1)
        dag.add_creg(cr2)
        node = dag.apply_operation_back(IfElseOp(expr.logic_not(cr1), body.copy(), None), qr, [])

        with self.assertRaisesRegex(DAGCircuitError, "does not span the same wires"):
            dag.substitute_node(
                node, IfElseOp(expr.logic_not(cr2), body.copy(), None), inplace=inplace
            )

    @data(True, False)
    def test_replace_switch_with_another(self, inplace):
        """Test that one `SwitchCaseOp` can be replaced with another."""
        case = QuantumCircuit(1)
        case.x(0)

        qr = QuantumRegister(1)
        cr1 = ClassicalRegister(2)
        cr2 = ClassicalRegister(2)
        dag = DAGCircuit()
        dag.add_qreg(qr)
        dag.add_creg(cr1)
        dag.add_creg(cr2)
        node = dag.apply_operation_back(
            SwitchCaseOp(expr.lift(cr1), [((1, 3), case.copy())]), qr, []
        )
        dag.substitute_node(
            node, SwitchCaseOp(expr.bit_and(cr1, 1), [(1, case.copy())]), inplace=inplace
        )

        expected = DAGCircuit()
        expected.add_qreg(qr)
        expected.add_creg(cr1)
        expected.add_creg(cr2)
        expected.apply_operation_back(
            SwitchCaseOp(expr.bit_and(cr1, 1), [(1, case.copy())]), qr, []
        )

        self.assertEqual(dag, expected)

    @data(True, False)
    def test_reject_replace_switch_with_other_resources(self, inplace):
        """Test that the resources in the `target` of a `SwitchCaseOp` are checked against those in
        the node to be replaced."""
        case = QuantumCircuit(1)
        case.x(0)

        qr = QuantumRegister(1)
        cr1 = ClassicalRegister(2)
        cr2 = ClassicalRegister(2)
        dag = DAGCircuit()
        dag.add_qreg(qr)
        dag.add_creg(cr1)
        dag.add_creg(cr2)
        node = dag.apply_operation_back(
            SwitchCaseOp(expr.lift(cr1), [((1, 3), case.copy())]), qr, []
        )

        with self.assertRaisesRegex(DAGCircuitError, "does not span the same wires"):
            dag.substitute_node(
                node, SwitchCaseOp(expr.lift(cr2), [((1, 3), case.copy())]), inplace=inplace
            )


class TestReplaceBlock(QiskitTestCase):
    """Test replacing a block of nodes in a DAG."""

    def setUp(self):
        super().setUp()
        self.qc = QuantumCircuit(2)
        self.qc.cx(0, 1)
        self.qc.h(1)
        self.qc.cx(0, 1)
        self.dag = circuit_to_dag(self.qc)

    def test_cycle_check(self):
        """Validate that by default introducing a cycle errors."""
        nodes = list(self.dag.topological_op_nodes())
        with self.assertRaises(DAGCircuitError):
            self.dag.replace_block_with_op(
                [nodes[0], nodes[-1]],
                CZGate(),
                {bit: idx for (idx, bit) in enumerate(self.dag.qubits)},
            )

    def test_empty(self):
        """Test that an empty node block raises."""
        with self.assertRaises(DAGCircuitError):
            self.dag.replace_block_with_op(
                [], CZGate(), {bit: idx for (idx, bit) in enumerate(self.dag.qubits)}
            )

    def test_single_node_block(self):
        """Test that a single node as the block works."""
        qr = QuantumRegister(2)
        dag = DAGCircuit()
        dag.add_qreg(qr)
        node = dag.apply_operation_back(HGate(), [qr[0]])
        new_node = dag.replace_block_with_op(
            [node], XGate(), {bit: idx for (idx, bit) in enumerate(dag.qubits)}
        )

        expected_dag = DAGCircuit()
        expected_dag.add_qreg(qr)
        expected_dag.apply_operation_back(XGate(), [qr[0]])

        self.assertEqual(expected_dag, dag)
        self.assertEqual(expected_dag.count_ops(), dag.count_ops())
        self.assertIsInstance(new_node.op, XGate)

    def test_replace_control_flow_block(self):
        """Test that we can replace a block of control-flow nodes with a single one."""
        body = QuantumCircuit(1)
        body.x(0)

        qr = QuantumRegister(1)
        cr1 = ClassicalRegister(3)
        cr2 = ClassicalRegister(3)
        dag = DAGCircuit()
        dag.add_qreg(qr)
        dag.add_creg(cr1)
        dag.add_creg(cr2)
        nodes = [
            dag.apply_operation_back(IfElseOp(expr.logic_not(cr1[0]), body.copy(), None), qr, []),
            dag.apply_operation_back(
                SwitchCaseOp(expr.lift(cr2), [((0, 1, 2, 3), body.copy())]), qr, []
            ),
        ]

        dag.replace_block_with_op(
            nodes,
            IfElseOp(expr.logic_or(expr.logic_not(cr1[0]), expr.less(cr2, 4)), body.copy(), None),
            {qr[0]: 0},
        )

        expected = DAGCircuit()
        expected.add_qreg(qr)
        expected.add_creg(cr1)
        expected.add_creg(cr2)
        expected.apply_operation_back(
            IfElseOp(expr.logic_or(expr.logic_not(cr1[0]), expr.less(cr2, 4)), body.copy(), None),
            qr,
            [],
        )

        self.assertEqual(dag, expected)


class TestDagProperties(QiskitTestCase):
    """Test the DAG properties."""

    def setUp(self):
        #       ┌───┐                ┌───┐
        # q0_0: ┤ H ├────────────────┤ X ├──────────
        #       └───┘                └─┬─┘     ┌───┐
        # q0_1: ───────────────────────┼───────┤ H ├
        #                 ┌───┐        │  ┌───┐└─┬─┘
        # q0_2: ──■───────┤ H ├────────┼──┤ T ├──■──
        #       ┌─┴─┐┌────┴───┴─────┐  │  └───┘
        # q0_3: ┤ X ├┤ U(0,0.1,0.2) ├──┼────────────
        #       └───┘└──────────────┘  │
        # q1_0: ───────────────────────■────────────
        #                              │
        # q1_1: ───────────────────────■────────────
        super().setUp()
        qr1 = QuantumRegister(4)
        qr2 = QuantumRegister(2)
        circ = QuantumCircuit(qr1, qr2)
        circ.h(qr1[0])
        circ.cx(qr1[2], qr1[3])
        circ.h(qr1[2])
        circ.t(qr1[2])
        circ.ch(qr1[2], qr1[1])
        circ.u(0.0, 0.1, 0.2, qr1[3])
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
        operations = {"h": 2, "t": 1, "u": 1, "cx": 1, "ch": 1, "ccx": 1}

        self.assertDictEqual(self.dag.count_ops(), operations)

    def test_dag_factors(self):
        """Test number of separable factors in circuit."""
        self.assertEqual(self.dag.num_tensor_factors(), 2)

    def test_dag_depth_empty(self):
        """Empty circuit DAG is zero depth"""
        q = QuantumRegister(5, "q")
        qc = QuantumCircuit(q)
        dag = circuit_to_dag(qc)
        self.assertEqual(dag.depth(), 0)

    def test_dag_idle_wires(self):
        """Test dag idle_wires."""
        wires = list(self.dag.idle_wires())
        self.assertEqual(len(wires), 0)
        wires = list(self.dag.idle_wires(["u", "cx"]))
        self.assertEqual(len(wires), 1)

    def test_dag_depth1(self):
        """Test DAG depth #1"""
        #       ┌───┐
        # q1_0: ┤ H ├──■────■─────────────────
        #       ├───┤  │  ┌─┴─┐
        # q1_1: ┤ H ├──┼──┤ X ├──■────────────
        #       ├───┤  │  └───┘  │  ┌───┐
        # q1_2: ┤ H ├──┼─────────┼──┤ X ├──■──
        #       ├───┤┌─┴─┐       │  └─┬─┘┌─┴─┐
        # q2_0: ┤ H ├┤ X ├───────┼────┼──┤ X ├
        #       ├───┤└─┬─┘     ┌─┴─┐  │  └───┘
        # q2_1: ┤ H ├──■───────┤ X ├──■───────
        #       └───┘          └───┘
        qr1 = QuantumRegister(3, "q1")
        qr2 = QuantumRegister(2, "q2")
        c = ClassicalRegister(5, "c")
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
        """Test barrier increases DAG depth"""

        #      ┌───┐                     ░
        # q_0: ┤ H ├──■──────────────────░────
        #      └───┘  │            ┌───┐ ░ ┌─┐
        # q_1: ───────┼────────────┤ X ├─░─┤M├
        #      ┌───┐  │  ┌───┐┌───┐└─┬─┘ ░ └╥┘
        # q_2: ┤ X ├──┼──┤ X ├┤ X ├──┼───░──╫─
        #      └───┘  │  └───┘└───┘  │   ░  ║
        # q_3: ───────┼──────────────┼───░──╫─
        #           ┌─┴─┐┌───┐       │   ░  ║
        # q_4: ─────┤ X ├┤ X ├───────■───░──╫─
        #           └───┘└───┘           ░  ║
        # c: 1/═════════════════════════════╩═
        #                                   0
        q = QuantumRegister(5, "q")
        c = ClassicalRegister(1, "c")
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
        """Test DAG depth for silly circuit."""

        #      ┌───┐       ░    ░       ┌─┐
        # q_0: ┤ H ├──■────░────░───────┤M├─────
        #      └───┘┌─┴─┐  ░    ░       └╥┘
        # q_1: ─────┤ X ├──■─────────────╫──────
        #           └───┘┌─┴─┐           ║
        # q_2: ──────────┤ X ├──■────────╫──────
        #                └───┘┌─┴─┐      ║
        # q_3: ───────────────┤ X ├──■───╫──────
        #                     └───┘┌─┴─┐ ║
        # q_4: ────────────────────┤ X ├─╫───■──
        #                          └───┘ ║ ┌─┴─┐
        # q_5: ──────────────────────────╫─┤ X ├
        #                                ║ └───┘
        # c: 1/══════════════════════════╩══════
        #                                0
        q = QuantumRegister(6, "q")
        c = ClassicalRegister(1, "c")
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


class TestConditional(QiskitTestCase):
    """Test the classical conditional gates."""

    def setUp(self):
        super().setUp()
        self.qreg = QuantumRegister(3, "q")
        self.creg = ClassicalRegister(2, "c")
        self.creg2 = ClassicalRegister(2, "c2")
        self.qubit0 = self.qreg[0]
        self.circuit = QuantumCircuit(self.qreg, self.creg, self.creg2)
        self.dag = None

    def test_creg_conditional(self):
        """Test consistency of conditional on classical register."""

        self.circuit.h(self.qreg[0]).c_if(self.creg, 1)
        self.dag = circuit_to_dag(self.circuit)
        gate_node = self.dag.gate_nodes()[0]
        self.assertEqual(gate_node.op, HGate())
        self.assertEqual(gate_node.qargs, (self.qreg[0],))
        self.assertEqual(gate_node.cargs, ())
        self.assertEqual(gate_node.op.condition, (self.creg, 1))
        self.assertEqual(
            sorted(self.dag._multi_graph.in_edges(gate_node._node_id)),
            sorted(
                [
                    (self.dag.input_map[self.qreg[0]]._node_id, gate_node._node_id, self.qreg[0]),
                    (self.dag.input_map[self.creg[0]]._node_id, gate_node._node_id, self.creg[0]),
                    (self.dag.input_map[self.creg[1]]._node_id, gate_node._node_id, self.creg[1]),
                ]
            ),
        )

        self.assertEqual(
            sorted(self.dag._multi_graph.out_edges(gate_node._node_id)),
            sorted(
                [
                    (gate_node._node_id, self.dag.output_map[self.qreg[0]]._node_id, self.qreg[0]),
                    (gate_node._node_id, self.dag.output_map[self.creg[0]]._node_id, self.creg[0]),
                    (gate_node._node_id, self.dag.output_map[self.creg[1]]._node_id, self.creg[1]),
                ]
            ),
        )

    def test_clbit_conditional(self):
        """Test consistency of conditional on single classical bit."""

        self.circuit.h(self.qreg[0]).c_if(self.creg[0], 1)
        self.dag = circuit_to_dag(self.circuit)
        gate_node = self.dag.gate_nodes()[0]
        self.assertEqual(gate_node.op, HGate())
        self.assertEqual(gate_node.qargs, (self.qreg[0],))
        self.assertEqual(gate_node.cargs, ())
        self.assertEqual(gate_node.op.condition, (self.creg[0], 1))
        self.assertEqual(
            sorted(self.dag._multi_graph.in_edges(gate_node._node_id)),
            sorted(
                [
                    (self.dag.input_map[self.qreg[0]]._node_id, gate_node._node_id, self.qreg[0]),
                    (self.dag.input_map[self.creg[0]]._node_id, gate_node._node_id, self.creg[0]),
                ]
            ),
        )

        self.assertEqual(
            sorted(self.dag._multi_graph.out_edges(gate_node._node_id)),
            sorted(
                [
                    (gate_node._node_id, self.dag.output_map[self.qreg[0]]._node_id, self.qreg[0]),
                    (gate_node._node_id, self.dag.output_map[self.creg[0]]._node_id, self.creg[0]),
                ]
            ),
        )


class TestSwapNodes(QiskitTestCase):
    """Test Swapping connected nodes."""

    def test_1q_swap_fully_connected(self):
        """Test swapping single qubit gates"""
        dag = DAGCircuit()
        qreg = QuantumRegister(1)
        dag.add_qreg(qreg)
        op_node0 = dag.apply_operation_back(RXGate(pi / 2), [qreg[0]], [])
        op_node1 = dag.apply_operation_back(RXGate(pi), [qreg[0]], [])
        dag.swap_nodes(op_node0, op_node1)

        expected = DAGCircuit()
        expected.add_qreg(qreg)
        expected.apply_operation_back(RXGate(pi), [qreg[0]], [])
        expected.apply_operation_back(RXGate(pi / 2), [qreg[0]], [])

        self.assertEqual(dag, expected)

    def test_2q_swap_fully_connected(self):
        """test swaping full connected 2q gates"""
        dag = DAGCircuit()
        qreg = QuantumRegister(2)
        dag.add_qreg(qreg)
        op_node0 = dag.apply_operation_back(CXGate(), qreg[[0, 1]], [])
        op_node1 = dag.apply_operation_back(CZGate(), qreg[[1, 0]], [])
        dag.swap_nodes(op_node0, op_node1)

        expected = DAGCircuit()
        expected.add_qreg(qreg)
        expected.apply_operation_back(CZGate(), qreg[[1, 0]], [])
        expected.apply_operation_back(CXGate(), qreg[[0, 1]], [])

        self.assertEqual(dag, expected)

    def test_2q_swap_partially_connected(self):
        """test swapping 2q partially connected gates"""
        dag = DAGCircuit()
        qreg = QuantumRegister(3)
        dag.add_qreg(qreg)
        op_node0 = dag.apply_operation_back(CXGate(), qreg[[1, 0]], [])
        op_node1 = dag.apply_operation_back(CZGate(), qreg[[1, 2]], [])
        dag.swap_nodes(op_node0, op_node1)

        expected = DAGCircuit()
        expected.add_qreg(qreg)
        expected.apply_operation_back(CZGate(), qreg[[1, 2]], [])
        expected.apply_operation_back(CXGate(), qreg[[1, 0]], [])
        self.assertEqual(dag, expected)

    def test_4q_swap_partially_connected(self):
        """test swapping 4q partially connected gates"""
        from qiskit.circuit.library.standard_gates import C3XGate

        dag = DAGCircuit()
        qreg = QuantumRegister(5)
        dag.add_qreg(qreg)
        op_node0 = dag.apply_operation_back(C3XGate(), qreg[[0, 1, 2, 3]], [])
        op_node1 = dag.apply_operation_back(C3XGate(), qreg[[0, 1, 2, 4]], [])
        dag.swap_nodes(op_node0, op_node1)

        expected = DAGCircuit()
        expected.add_qreg(qreg)
        expected.apply_operation_back(C3XGate(), qreg[[0, 1, 2, 4]], [])
        expected.apply_operation_back(C3XGate(), qreg[[0, 1, 2, 3]], [])
        self.assertEqual(dag, expected)

    def test_2q_swap_non_connected_raises(self):
        """test swapping 2q non-connected gates raises an exception"""
        dag = DAGCircuit()
        qreg = QuantumRegister(4)
        dag.add_qreg(qreg)
        op_node0 = dag.apply_operation_back(CXGate(), qreg[[0, 2]], [])
        op_node1 = dag.apply_operation_back(CZGate(), qreg[[1, 3]], [])
        self.assertRaises(DAGCircuitError, dag.swap_nodes, op_node0, op_node1)

    def test_2q_swap_partially_connected_pre_spectators(self):
        """test swapping 2q partially connected gates when in the presence of pre
        spectator ops."""
        dag = DAGCircuit()
        qreg = QuantumRegister(5)
        dag.add_qreg(qreg)
        dag.apply_operation_back(XGate(), qreg[[0]])
        op_node0 = dag.apply_operation_back(CXGate(), qreg[[1, 0]], [])
        op_node1 = dag.apply_operation_back(CZGate(), qreg[[1, 2]], [])
        dag.swap_nodes(op_node0, op_node1)

        expected = DAGCircuit()
        expected.add_qreg(qreg)
        expected.apply_operation_back(XGate(), qreg[[0]])
        expected.apply_operation_back(CZGate(), qreg[[1, 2]], [])
        expected.apply_operation_back(CXGate(), qreg[[1, 0]], [])
        self.assertEqual(dag, expected)

    def test_2q_swap_partially_connected_prepost_spectators(self):
        """test swapping 2q partially connected gates when in the presence of pre
        and post spectator ops."""
        dag = DAGCircuit()
        qreg = QuantumRegister(5)
        dag.add_qreg(qreg)
        dag.apply_operation_back(XGate(), qreg[[0]])
        op_node0 = dag.apply_operation_back(CXGate(), qreg[[1, 0]], [])
        op_node1 = dag.apply_operation_back(CZGate(), qreg[[1, 2]], [])
        dag.apply_operation_back(CZGate(), qreg[[1, 2]], [])
        dag.swap_nodes(op_node0, op_node1)

        expected = DAGCircuit()
        expected.add_qreg(qreg)
        expected.apply_operation_back(XGate(), qreg[[0]])
        expected.apply_operation_back(CZGate(), qreg[[1, 2]], [])
        expected.apply_operation_back(CXGate(), qreg[[1, 0]], [])
        expected.apply_operation_back(CZGate(), qreg[[1, 2]], [])
        self.assertEqual(dag, expected)


class TestDagCausalCone(QiskitTestCase):
    """Test `get_causal_node` function"""

    def test_causal_cone_regular_circuit(self):
        """Test causal cone with a regular circuit"""

        # q_0: ───────■─────────────
        #             │
        # q_1: ──■────┼───■─────────
        #      ┌─┴─┐  │   │
        # q_2: ┤ X ├──┼───┼──■──────
        #      └───┘┌─┴─┐ │  │
        # q_3: ─────┤ X ├─┼──┼───■──
        #           └───┘ │  │ ┌─┴─┐
        # q_4: ───────────■──■─┤ X ├
        #                      └───┘
        # c: 5/═════════════════════

        dag = DAGCircuit()
        qreg = QuantumRegister(5)
        creg = ClassicalRegister(5)
        dag.add_qreg(qreg)
        dag.add_creg(creg)
        dag.apply_operation_back(CXGate(), qreg[[1, 2]], [])
        dag.apply_operation_back(CXGate(), qreg[[0, 3]], [])
        dag.apply_operation_back(CZGate(), qreg[[1, 4]], [])
        dag.apply_operation_back(CZGate(), qreg[[2, 4]], [])
        dag.apply_operation_back(CXGate(), qreg[[3, 4]], [])

        # Get causal cone of qubit at index 0
        result = dag.quantum_causal_cone(qreg[0])

        # Expected result
        expected = set(qreg[[0, 3]])
        self.assertEqual(result, expected)

    def test_causal_cone_invalid_qubit(self):
        """Test causal cone with invalid qubit"""

        # q_0: ───────■─────────────
        #             │
        # q_1: ──■────┼───■─────────
        #      ┌─┴─┐  │   │
        # q_2: ┤ X ├──┼───┼──■──────
        #      └───┘┌─┴─┐ │  │
        # q_3: ─────┤ X ├─┼──┼───■──
        #           └───┘ │  │ ┌─┴─┐
        # q_4: ───────────■──■─┤ X ├
        #                      └───┘
        # c: 5/═════════════════════

        dag = DAGCircuit()
        qreg = QuantumRegister(5)
        creg = ClassicalRegister(5)
        dag.add_qreg(qreg)
        dag.add_creg(creg)
        dag.apply_operation_back(CXGate(), qreg[[1, 2]], [])
        dag.apply_operation_back(CXGate(), qreg[[0, 3]], [])
        dag.apply_operation_back(CZGate(), qreg[[1, 4]], [])
        dag.apply_operation_back(CZGate(), qreg[[2, 4]], [])
        dag.apply_operation_back(CXGate(), qreg[[3, 4]], [])

        # Raise error due to invalid index
        self.assertRaises(DAGCircuitError, dag.quantum_causal_cone, Qubit())

    def test_causal_cone_no_neighbor(self):
        """Test causal cone with no neighbor"""

        # q_0: ───────────

        # q_1: ──■───■────
        #      ┌─┴─┐ │
        # q_2: ┤ X ├─┼──■─
        #      ├───┤ │  │
        # q_3: ┤ X ├─┼──┼─
        #      └───┘ │  │
        # q_4: ──────■──■─

        # c: 5/═══════════

        dag = DAGCircuit()
        qreg = QuantumRegister(5)
        creg = ClassicalRegister(5)
        dag.add_qreg(qreg)
        dag.add_creg(creg)
        dag.apply_operation_back(CXGate(), qreg[[1, 2]], [])
        dag.apply_operation_back(CZGate(), qreg[[1, 4]], [])
        dag.apply_operation_back(CZGate(), qreg[[2, 4]], [])
        dag.apply_operation_back(XGate(), qreg[[3]], [])

        # Get causal cone of Qubit at index 3.
        result = dag.quantum_causal_cone(qreg[3])
        # Expect only a set with Qubit at index 3
        expected = set(qreg[[3]])
        self.assertEqual(result, expected)

    def test_causal_cone_empty_circuit(self):
        """Test causal cone for circuit with no operations"""
        dag = DAGCircuit()
        qreg = QuantumRegister(5)
        creg = ClassicalRegister(5)
        dag.add_qreg(qreg)
        dag.add_creg(creg)

        # Get causal cone of qubit at index 4
        result = dag.quantum_causal_cone(qreg[4])
        # Expect only a set with Qubit at index 4
        expected = set(qreg[[4]])

        self.assertEqual(result, expected)

    def test_causal_cone_barriers(self):
        """Test causal cone for circuit with barriers"""

        #       ┌───┐      ░               ░
        # q1_0: ┤ X ├──■───░───────────────░───────────
        #       └───┘┌─┴─┐ ░               ░
        # q1_1: ─────┤ X ├─░───────■───■───░───────────
        #            └───┘ ░ ┌───┐ │   │   ░ ┌───┐
        # q1_2: ───────────░─┤ H ├─■───┼───░─┤ Y ├─────
        #                  ░ └───┘   ┌─┴─┐ ░ └─┬─┘┌───┐
        # q1_3: ───────────░─────────┤ Y ├─░───■──┤ X ├
        #                  ░         └───┘ ░      └─┬─┘
        # q1_4: ───────────░───────────────░────────■──
        #                  ░               ░

        # Build the circuit:
        qreg = QuantumRegister(5)
        qc = QuantumCircuit(qreg)
        qc.x(0)
        qc.cx(0, 1)
        qc.barrier()

        qc.h(2)
        qc.cz(2, 1)
        qc.cy(1, 3)
        qc.barrier()

        qc.cy(3, 2)
        qc.cx(4, 3)

        # Transform into a dag:
        dag = circuit_to_dag(qc)

        # Compute result:
        result = dag.quantum_causal_cone(qreg[1])
        # Expected:
        expected = {qreg[0], qreg[1], qreg[2], qreg[3]}

        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
