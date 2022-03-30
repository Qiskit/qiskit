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

from collections import Counter
import unittest

from ddt import ddt, data

import retworkx as rx
from numpy import pi

from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGInNode, DAGOutNode
from qiskit.circuit import QuantumRegister
from qiskit.circuit import ClassicalRegister, Clbit
from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.circuit import Measure
from qiskit.circuit import Reset
from qiskit.circuit import Delay
from qiskit.circuit import Gate, Instruction
from qiskit.circuit import Parameter
from qiskit.circuit.library.standard_gates.i import IGate
from qiskit.circuit.library.standard_gates.h import HGate
from qiskit.circuit.library.standard_gates.x import CXGate
from qiskit.circuit.library.standard_gates.z import CZGate
from qiskit.circuit.library.standard_gates.x import XGate
from qiskit.circuit.library.standard_gates.y import YGate
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

        node_cond_bits = set(node.op.condition[0][:] if node.op.condition is not None else [])
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
            ``cregs`` before the comparison."""
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
            ``clbits`` before the comparison."""
        if excluding is None:
            excluding = set()
        self.assertEqual(self.dag.clbits, list(b for b in clbits if b not in excluding))

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

    def test_copy_circuit_metadata(self):
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
        h_node = self.dag.apply_operation_back(h_gate, [self.qubit2], [])

        self.assertEqual(h_node.qargs, [self.qubit2])
        self.assertEqual(h_node.cargs, [])
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

        meas_gate = Measure()
        meas_gate.condition = (new_creg, 0)
        meas_node = self.dag.apply_operation_back(meas_gate, [self.qubit0], [self.clbit0])

        self.assertEqual(meas_node.qargs, [self.qubit0])
        self.assertEqual(meas_node.cargs, [self.clbit0])
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

        meas_gate = Measure()
        meas_gate.condition = self.condition
        meas_node = self.dag.apply_operation_back(meas_gate, [self.qubit1], [self.clbit1])

        self.assertEqual(meas_node.qargs, [self.qubit1])
        self.assertEqual(meas_node.cargs, [self.clbit1])
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

        self.dag.apply_operation_back(Measure(), [self.qubit1, self.clbit1], [])
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
        self.dag.apply_operation_back(Measure(), [self.qubit1, self.clbit1], [])
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
        self.dag.apply_operation_back(Measure(), [self.qubit1, self.clbit1], [])

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

    def test_is_predecessor(self):
        """The method dag.is_predecessor(A, B) checks if node B is a predecessor of A"""

        self.dag.apply_operation_back(Measure(), [self.qubit1, self.clbit1], [])
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
            ("cx", [self.qubit0, self.qubit1]),
            ("h", [self.qubit0]),
            qr[2],
            ("cx", [self.qubit2, self.qubit1]),
            ("cx", [self.qubit0, self.qubit2]),
            ("h", [self.qubit2]),
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
            ("cx", [self.qubit0, self.qubit1]),
            ("h", [self.qubit0]),
            ("cx", [self.qubit2, self.qubit1]),
            ("cx", [self.qubit0, self.qubit2]),
            ("h", [self.qubit2]),
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
            ("h", [self.qubit0]),
            ("cx", [self.qubit2, self.qubit1]),
            ("cx", [self.qubit0, self.qubit2]),
            ("h", [self.qubit2]),
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
                    [[self.qubit2, self.qubit1], [self.qubit1, self.qubit2]], [x.qargs for x in run]
                )
            elif run[0].op.name == "h":
                self.assertEqual(len(run), 1)
                self.assertEqual(["h"], [x.op.name for x in run])
                self.assertEqual([[self.qubit2]], [x.qargs for x in run])
            elif run[0].op.name == "u1":
                self.assertEqual(len(run), 3)
                self.assertEqual(["u1"] * 3, [x.op.name for x in run])
                self.assertEqual(
                    [[self.qubit0], [self.qubit0], [self.qubit0]], [x.qargs for x in run]
                )
            else:
                self.fail("Unknown run encountered")

    def test_dag_collect_runs_start_with_conditional(self):
        """Test collect runs with a conditional at the start of the run."""
        h_gate = HGate()
        h_gate.condition = self.condition
        self.dag.apply_operation_back(h_gate, [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        collected_runs = self.dag.collect_runs(["h"])
        self.assertEqual(len(collected_runs), 1)
        run = collected_runs.pop()
        self.assertEqual(len(run), 2)
        self.assertEqual(["h", "h"], [x.op.name for x in run])
        self.assertEqual([[self.qubit0], [self.qubit0]], [x.qargs for x in run])

    def test_dag_collect_runs_conditional_in_middle(self):
        """Test collect_runs with a conditional in the middle of a run."""
        h_gate = HGate()
        h_gate.condition = self.condition
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(h_gate, [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        collected_runs = self.dag.collect_runs(["h"])
        # Should return 2 single h gate runs (1 before condition, 1 after)
        self.assertEqual(len(collected_runs), 2)
        for run in collected_runs:
            self.assertEqual(len(run), 1)
            self.assertEqual(["h"], [x.op.name for x in run])
            self.assertEqual([[self.qubit0]], [x.qargs for x in run])

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
                self.assertEqual([[self.qubit2]], [x.qargs for x in run])
            elif run[0].op.name == "u1":
                self.assertEqual(len(run), 3)
                self.assertEqual(["u1"] * 3, [x.op.name for x in run])
                self.assertEqual(
                    [[self.qubit0], [self.qubit0], [self.qubit0]], [x.qargs for x in run]
                )
            else:
                self.fail("Unknown run encountered")

    def test_dag_collect_1q_runs_start_with_conditional(self):
        """Test collect 1q runs with a conditional at the start of the run."""
        self.dag.apply_operation_back(Reset(), [self.qubit0])
        self.dag.apply_operation_back(Delay(100), [self.qubit0])
        h_gate = HGate()
        h_gate.condition = self.condition
        self.dag.apply_operation_back(h_gate, [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        collected_runs = self.dag.collect_1q_runs()
        self.assertEqual(len(collected_runs), 1)
        run = collected_runs.pop()
        self.assertEqual(len(run), 2)
        self.assertEqual(["h", "h"], [x.op.name for x in run])
        self.assertEqual([[self.qubit0], [self.qubit0]], [x.qargs for x in run])

    def test_dag_collect_1q_runs_conditional_in_middle(self):
        """Test collect_1q_runs with a conditional in the middle of a run."""
        self.dag.apply_operation_back(Reset(), [self.qubit0])
        self.dag.apply_operation_back(Delay(100), [self.qubit0])
        h_gate = HGate()
        h_gate.condition = self.condition
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(h_gate, [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        collected_runs = self.dag.collect_1q_runs()
        # Should return 2 single h gate runs (1 before condition, 1 after)
        self.assertEqual(len(collected_runs), 2)
        for run in collected_runs:
            self.assertEqual(len(run), 1)
            self.assertEqual(["h"], [x.op.name for x in run])
            self.assertEqual([[self.qubit0]], [x.qargs for x in run])

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
                self.assertEqual([[self.qubit0]] * 3, [x.qargs for x in run])
            elif run[0].op.name == "u1":
                self.assertEqual(len(run), 3)
                self.assertEqual(["u1", "u1", "h"], [x.op.name for x in run])
                self.assertEqual([[self.qubit1]] * 3, [x.qargs for x in run])
            elif run[0].op.name == "x":
                self.assertEqual(len(run), 2)
                self.assertEqual(["x", "x"], [x.op.name for x in run])
                self.assertEqual([[self.qubit1]] * 2, [x.qargs for x in run])
            elif run[0].op.name == "y":
                self.assertEqual(len(run), 2)
                self.assertEqual(["y", "y"], [x.op.name for x in run])
                self.assertEqual([[self.qubit0]] * 2, [x.qargs for x in run])
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

    def test_dag_from_networkx(self):
        """Test DAG from networkx creates an expected DAGCircuit object."""
        from copy import deepcopy
        from collections import OrderedDict

        nx_graph = self.dag1.to_networkx()
        from_nx_dag = DAGCircuit.from_networkx(nx_graph)

        # to_/from_networkx does not preserve Registers or bit indexing,
        # so remove them from reference DAG.
        dag = deepcopy(self.dag1)
        dag.qregs = OrderedDict()
        dag.cregs = OrderedDict()
        dag.qubits = from_nx_dag.qubits
        dag.clbits = from_nx_dag.clbits

        self.assertEqual(dag, from_nx_dag)

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

    def test_raise_if_substituting_dag_modifies_its_conditional(self):
        """Verify that we raise if the input dag modifies any of the bits in node.op.condition."""

        # Our unroller's rely on substitute_node_with_dag carrying the condition
        # from the node over to the input dag, which it does by making every
        # node in the input dag conditioned over the same bits. However, if the
        # input dag e.g. measures to one of those bits, the behavior of the
        # remainder of the DAG would be different, so detect and raise in that
        # case.

        instr = Instruction("opaque", 1, 1, [])
        instr.condition = self.condition
        instr_node = self.dag.apply_operation_back(instr, [self.qubit0], [self.clbit1])

        sub_dag = DAGCircuit()
        sub_qr = QuantumRegister(1, "sqr")
        sub_cr = ClassicalRegister(1, "scr")
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

        replacement_node = dag.substitute_node(node_to_be_replaced, CZGate(), inplace=inplace)

        raise_if_dagcircuit_invalid(dag)
        self.assertEqual(replacement_node.op.name, "cz")
        self.assertEqual(replacement_node.qargs, [qr[1], qr[0]])
        self.assertEqual(replacement_node.cargs, [])
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
        dag.replace_block_with_op(
            [node], XGate(), {bit: idx for (idx, bit) in enumerate(dag.qubits)}
        )

        expected_dag = DAGCircuit()
        expected_dag.add_qreg(qr)
        expected_dag.apply_operation_back(XGate(), [qr[0]])

        self.assertEqual(expected_dag, dag)
        self.assertEqual(expected_dag.count_ops(), dag.count_ops())


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
        self.assertEqual(gate_node.qargs, [self.qreg[0]])
        self.assertEqual(gate_node.cargs, [])
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
        self.assertEqual(gate_node.qargs, [self.qreg[0]])
        self.assertEqual(gate_node.cargs, [])
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


class TestDAGDeprecations(QiskitTestCase):
    """Test DAG deprecations"""

    def test_DAGNode_deprecations(self):
        """Test DAGNode deprecations."""
        from qiskit.dagcircuit import DAGNode

        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        with self.assertWarnsRegex(DeprecationWarning, "deprecated"):
            op_node = DAGNode(type="op", op=HGate(), qargs=[qr[0]], cargs=[cr[0]])
        with self.assertWarnsRegex(DeprecationWarning, "deprecated"):
            in_node = DAGNode(type="in", wire=qr[0])
        with self.assertWarnsRegex(DeprecationWarning, "deprecated"):
            out_node = DAGNode(type="out", wire=cr[0])
        with self.assertWarnsRegex(DeprecationWarning, "deprecated"):
            _ = op_node.type
        with self.assertWarnsRegex(DeprecationWarning, "deprecated"):
            _ = op_node.op
        with self.assertWarnsRegex(DeprecationWarning, "deprecated"):
            _ = op_node.qargs
        with self.assertWarnsRegex(DeprecationWarning, "deprecated"):
            _ = op_node.cargs
        with self.assertWarnsRegex(DeprecationWarning, "deprecated"):
            _ = in_node.wire
        with self.assertWarnsRegex(DeprecationWarning, "deprecated"):
            _ = out_node.wire


if __name__ == "__main__":
    unittest.main()
