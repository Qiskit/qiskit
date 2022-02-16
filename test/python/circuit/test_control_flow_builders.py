# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test operations on the builder interfaces for control flow in dynamic QuantumCircuits."""

import copy
import math

import ddt

from qiskit.circuit import (
    ClassicalRegister,
    Clbit,
    Measure,
    Parameter,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit.controlflow import ForLoopOp, IfElseOp, WhileLoopOp, BreakLoopOp, ContinueLoopOp
from qiskit.circuit.controlflow.builder import ControlFlowBuilderBlock
from qiskit.circuit.controlflow.if_else import IfElsePlaceholder
from qiskit.circuit.exceptions import CircuitError
from qiskit.test import QiskitTestCase


class SentinelException(Exception):
    """An exception that we know was raised deliberately."""


@ddt.ddt
class TestControlFlowBuilders(QiskitTestCase):
    """Test that the control-flow builder interfaces work, and manage resources correctly."""

    def assertCircuitsEquivalent(self, a, b):
        """Assert that two circuits (``a`` and ``b``) contain all the same qubits and clbits, and
        then have the same instructions in order, recursing into nested control-flow constructs.

        Relying on ``QuantumCircuit.__eq__`` doesn't work reliably in all cases here, because we
        don't care about the order that the builder interface chooses for resources in the inner
        blocks.  This order is non-deterministic, because internally it uses sets for efficiency,
        and the order of iteration through a set is dependent on the hash seed.  Instead, we just
        need to be a bit more explicit about what we care about.  This isn't a full method for
        comparing if two circuits are equivalent, but for the restricted cases used in these tests,
        where we deliberately construct the expected result to be equal in the good case, it should
        test all that is needed.
        """

        self.assertIsInstance(a, QuantumCircuit)
        self.assertIsInstance(b, QuantumCircuit)

        # For our purposes here, we don't care about the order bits were added.
        self.assertEqual(set(a.qubits), set(b.qubits))
        self.assertEqual(set(a.clbits), set(b.clbits))
        self.assertEqual(set(a.qregs), set(b.qregs))
        self.assertEqual(set(a.cregs), set(b.cregs))
        self.assertEqual(len(a.data), len(b.data))

        for (a_op, a_qubits, a_clbits), (b_op, b_qubits, b_clbits) in zip(a.data, b.data):
            # Make sure that the operations are the same.
            self.assertEqual(type(a_op), type(b_op))
            self.assertEqual(hasattr(a_op, "condition"), hasattr(b_op, "condition"))
            if hasattr(a_op, "condition") and not isinstance(a_op, (IfElseOp, WhileLoopOp)):
                self.assertEqual(a_op.condition, b_op.condition)
            self.assertEqual(hasattr(a_op, "label"), hasattr(b_op, "label"))
            if hasattr(a_op, "condition"):
                self.assertEqual(a_op.label, b_op.label)
            # If it's a block op, we don't care what order the resources are specified in.
            if isinstance(a_op, WhileLoopOp):
                self.assertEqual(set(a_qubits), set(b_qubits))
                self.assertEqual(set(a_clbits), set(b_clbits))
                self.assertEqual(a_op.condition, b_op.condition)
                self.assertCircuitsEquivalent(a_op.blocks[0], b_op.blocks[0])
            elif isinstance(a_op, ForLoopOp):
                self.assertEqual(set(a_qubits), set(b_qubits))
                self.assertEqual(set(a_clbits), set(b_clbits))
                a_indexset, a_loop_parameter, a_body = a_op.params
                b_indexset, b_loop_parameter, b_body = b_op.params
                self.assertEqual(a_loop_parameter is None, b_loop_parameter is None)
                self.assertEqual(a_indexset, b_indexset)
                if a_loop_parameter is not None:
                    a_body = a_body.assign_parameters({a_loop_parameter: b_loop_parameter})
                self.assertCircuitsEquivalent(a_body, b_body)
            elif isinstance(a_op, IfElseOp):
                self.assertEqual(set(a_qubits), set(b_qubits))
                self.assertEqual(set(a_clbits), set(b_clbits))
                self.assertEqual(a_op.condition, b_op.condition)
                self.assertEqual(len(a_op.blocks), len(b_op.blocks))
                self.assertCircuitsEquivalent(a_op.blocks[0], b_op.blocks[0])
                if len(a_op.blocks) > 1:
                    self.assertCircuitsEquivalent(a_op.blocks[1], b_op.blocks[1])
            elif isinstance(a_op, (BreakLoopOp, ContinueLoopOp)):
                self.assertEqual(set(a_qubits), set(b_qubits))
                self.assertEqual(set(a_clbits), set(b_clbits))
            else:
                # For any other op, we care that the resources are the same, and in the same order,
                # but we don't mind what sort of iterable they're contained in.
                self.assertEqual(tuple(a_qubits), tuple(b_qubits))
                self.assertEqual(tuple(a_clbits), tuple(b_clbits))
                self.assertEqual(a_op, b_op)

    def test_if_simple(self):
        """Test a simple if statement builds correctly, in the midst of other instructions."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]

        test = QuantumCircuit(qubits, clbits)
        test.h(0)
        test.measure(0, 0)
        with test.if_test((clbits[0], 0)):
            test.x(0)
        test.h(0)
        test.measure(0, 1)
        with test.if_test((clbits[1], 0)):
            test.h(1)
            test.cx(1, 0)

        if_true0 = QuantumCircuit([qubits[0], clbits[0]])
        if_true0.x(qubits[0])

        if_true1 = QuantumCircuit([qubits[0], qubits[1], clbits[1]])
        if_true1.h(qubits[1])
        if_true1.cx(qubits[1], qubits[0])

        expected = QuantumCircuit(qubits, clbits)
        expected.h(qubits[0])
        expected.measure(qubits[0], clbits[0])
        expected.if_test((clbits[0], 0), if_true0, [qubits[0]], [clbits[0]])
        expected.h(qubits[0])
        expected.measure(qubits[0], clbits[1])
        expected.if_test((clbits[1], 0), if_true1, [qubits[0], qubits[1]], [clbits[1]])

        self.assertCircuitsEquivalent(test, expected)

    def test_if_register(self):
        """Test a simple if statement builds correctly, when using a register as the condition.
        This requires the builder to unpack all the bits from the register to use as resources."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)

        test = QuantumCircuit(qr, cr)
        test.measure(qr, cr)
        with test.if_test((cr, 0)):
            test.x(0)

        if_true0 = QuantumCircuit([qr[0]], cr)
        if_true0.x(qr[0])

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr, cr)
        expected.if_test((cr, 0), if_true0, [qr[0]], [cr[:]])

        self.assertCircuitsEquivalent(test, expected)

    def test_register_condition_in_nested_block(self):
        """Test that nested blocks can use registers of the outermost circuits as conditions, and
        they get propagated through all the blocks."""

        qr = QuantumRegister(2)
        clbits = [Clbit(), Clbit(), Clbit()]
        cr1 = ClassicalRegister(3)
        # Try aliased classical registers as well, to catch potential overlap bugs.
        cr2 = ClassicalRegister(bits=clbits[:2])
        cr3 = ClassicalRegister(bits=clbits[1:])
        cr4 = ClassicalRegister(bits=clbits)

        with self.subTest("for/if"):
            test = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            with test.for_loop(range(3)):
                with test.if_test((cr1, 0)):
                    test.x(0)
                with test.if_test((cr2, 0)):
                    test.y(0)
                with test.if_test((cr3, 0)):
                    test.z(0)

            true_body1 = QuantumCircuit([qr[0]], cr1)
            true_body1.x(0)
            true_body2 = QuantumCircuit([qr[0]], cr2)
            true_body2.y(0)
            true_body3 = QuantumCircuit([qr[0]], cr3)
            true_body3.z(0)

            for_body = QuantumCircuit([qr[0]], clbits, cr1, cr2, cr3)  # but not cr4.
            for_body.if_test((cr1, 0), true_body1, [qr[0]], cr1)
            for_body.if_test((cr2, 0), true_body2, [qr[0]], cr2)
            for_body.if_test((cr3, 0), true_body3, [qr[0]], cr3)

            expected = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            expected.for_loop(range(3), None, for_body, [qr[0]], clbits + list(cr1))

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("for/while"):
            test = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            with test.for_loop(range(3)):
                with test.while_loop((cr1, 0)):
                    test.x(0)
                with test.while_loop((cr2, 0)):
                    test.y(0)
                with test.while_loop((cr3, 0)):
                    test.z(0)

            while_body1 = QuantumCircuit([qr[0]], cr1)
            while_body1.x(0)
            while_body2 = QuantumCircuit([qr[0]], cr2)
            while_body2.y(0)
            while_body3 = QuantumCircuit([qr[0]], cr3)
            while_body3.z(0)

            for_body = QuantumCircuit([qr[0]], clbits, cr1, cr2, cr3)
            for_body.while_loop((cr1, 0), while_body1, [qr[0]], cr1)
            for_body.while_loop((cr2, 0), while_body2, [qr[0]], cr2)
            for_body.while_loop((cr3, 0), while_body3, [qr[0]], cr3)

            expected = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            expected.for_loop(range(3), None, for_body, [qr[0]], clbits + list(cr1))

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("if/c_if"):
            test = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            with test.if_test((cr1, 0)):
                test.x(0).c_if(cr2, 0)
                test.z(0).c_if(cr3, 0)

            true_body = QuantumCircuit([qr[0]], cr1, cr2, cr3)
            true_body.x(0).c_if(cr2, 0)
            true_body.z(0).c_if(cr3, 0)

            expected = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            expected.if_test((cr1, 0), true_body, [qr[0]], clbits + list(cr1))

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("while/else/c_if"):
            test = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            with test.while_loop((cr1, 0)):
                with test.if_test((cr2, 0)) as else_:
                    test.x(0).c_if(cr3, 0)
                with else_:
                    test.z(0).c_if(cr4, 0)

            true_body = QuantumCircuit([qr[0]], cr2, cr3, cr4)
            true_body.x(0).c_if(cr3, 0)
            false_body = QuantumCircuit([qr[0]], cr2, cr3, cr4)
            false_body.z(0).c_if(cr4, 0)

            while_body = QuantumCircuit([qr[0]], cr1, cr2, cr3, cr4)
            while_body.if_else((cr2, 0), true_body, false_body, [qr[0]], clbits)

            expected = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            expected.while_loop((cr1, 0), while_body, [qr[0]], clbits + list(cr1))

            self.assertCircuitsEquivalent(test, expected)

    def test_if_else_simple(self):
        """Test a simple if/else statement builds correctly, in the midst of other instructions.
        This test has paired if and else blocks the same natural width."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]

        test = QuantumCircuit(qubits, clbits)
        test.h(0)
        test.measure(0, 0)
        with test.if_test((clbits[0], 0)) as else_:
            test.x(0)
        with else_:
            test.z(0)
        test.h(0)
        test.measure(0, 1)
        with test.if_test((clbits[1], 0)) as else_:
            test.h(1)
            test.cx(1, 0)
        with else_:
            test.h(0)
            test.h(1)

        # Both the if and else blocks in this circuit are the same natural width to begin with.
        if_true0 = QuantumCircuit([qubits[0], clbits[0]])
        if_true0.x(qubits[0])
        if_false0 = QuantumCircuit([qubits[0], clbits[0]])
        if_false0.z(qubits[0])

        if_true1 = QuantumCircuit([qubits[0], qubits[1], clbits[1]])
        if_true1.h(qubits[1])
        if_true1.cx(qubits[1], qubits[0])
        if_false1 = QuantumCircuit([qubits[0], qubits[1], clbits[1]])
        if_false1.h(qubits[0])
        if_false1.h(qubits[1])

        expected = QuantumCircuit(qubits, clbits)
        expected.h(qubits[0])
        expected.measure(qubits[0], clbits[0])
        expected.if_else((clbits[0], 0), if_true0, if_false0, [qubits[0]], [clbits[0]])
        expected.h(qubits[0])
        expected.measure(qubits[0], clbits[1])
        expected.if_else((clbits[1], 0), if_true1, if_false1, [qubits[0], qubits[1]], [clbits[1]])

        self.assertCircuitsEquivalent(test, expected)

    def test_if_else_resources_expand_true_superset_false(self):
        """Test that the resources of the if and else bodies come out correctly if the true body
        needs a superset of the resources of the false body."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]

        test = QuantumCircuit(qubits, clbits)
        test.h(0)
        test.measure(0, 0)
        with test.if_test((clbits[0], 0)) as else_:
            test.x(0)
            test.measure(1, 1)
        with else_:
            test.z(0)

        if_true0 = QuantumCircuit(qubits, clbits)
        if_true0.x(qubits[0])
        if_true0.measure(qubits[1], clbits[1])
        # The false body doesn't actually use qubits[1] or clbits[1], but it still needs to contain
        # them so the bodies match.
        if_false0 = QuantumCircuit(qubits, clbits)
        if_false0.z(qubits[0])

        expected = QuantumCircuit(qubits, clbits)
        expected.h(qubits[0])
        expected.measure(qubits[0], clbits[0])
        expected.if_else((clbits[0], 0), if_true0, if_false0, qubits, clbits)

        self.assertCircuitsEquivalent(test, expected)

    def test_if_else_resources_expand_false_superset_true(self):
        """Test that the resources of the if and else bodies come out correctly if the false body
        needs a superset of the resources of the true body.  This requires that the manager
        correctly adds resources to the true body after it has been created."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]

        test = QuantumCircuit(qubits, clbits)
        test.h(0)
        test.measure(0, 0)
        with test.if_test((clbits[0], 0)) as else_:
            test.x(0)
        with else_:
            test.z(0)
            test.measure(1, 1)

        # The true body doesn't actually use qubits[1] or clbits[1], but it still needs to contain
        # them so the bodies match.
        if_true0 = QuantumCircuit(qubits, clbits)
        if_true0.x(qubits[0])
        if_false0 = QuantumCircuit(qubits, clbits)
        if_false0.z(qubits[0])
        if_false0.measure(qubits[1], clbits[1])

        expected = QuantumCircuit(qubits, clbits)
        expected.h(qubits[0])
        expected.measure(qubits[0], clbits[0])
        expected.if_else((clbits[0], 0), if_true0, if_false0, qubits, clbits)

        self.assertCircuitsEquivalent(test, expected)

    def test_if_else_resources_expand_true_false_symmetric_difference(self):
        """Test that the resources of the if and else bodies come out correctly if the sets of
        resources for the true body and the false body have some overlap, but neither is a subset of
        the other.  This tests that the flow of resources from block to block is simultaneously
        bidirectional."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]

        test = QuantumCircuit(qubits, clbits)
        test.h(0)
        test.measure(0, 0)
        with test.if_test((clbits[0], 0)) as else_:
            test.x(0)
        with else_:
            test.z(1)

        # The true body doesn't use qubits[1] and the false body doesn't use qubits[0].
        if_true0 = QuantumCircuit(qubits, [clbits[0]])
        if_true0.x(qubits[0])
        if_false0 = QuantumCircuit(qubits, [clbits[0]])
        if_false0.z(qubits[1])

        expected = QuantumCircuit(qubits, clbits)
        expected.h(qubits[0])
        expected.measure(qubits[0], clbits[0])
        expected.if_else((clbits[0], 0), if_true0, if_false0, qubits, [clbits[0]])

        self.assertCircuitsEquivalent(test, expected)

    def test_if_else_empty_branches(self):
        """Test that the context managers can cope with a body being empty."""
        qubits = [Qubit()]
        clbits = [Clbit()]

        cond = (clbits[0], 0)
        test = QuantumCircuit(qubits, clbits)
        # Sole empty if.
        with test.if_test(cond):
            pass
        # Normal if with an empty else body.
        with test.if_test(cond) as else_:
            test.x(0)
        with else_:
            pass
        # Empty if with a normal else body.
        with test.if_test(cond) as else_:
            pass
        with else_:
            test.x(0)
        # Both empty.
        with test.if_test(cond) as else_:
            pass
        with else_:
            pass

        empty_with_qubit = QuantumCircuit([qubits[0], clbits[0]])
        empty = QuantumCircuit([clbits[0]])
        only_x = QuantumCircuit([qubits[0], clbits[0]])
        only_x.x(qubits[0])

        expected = QuantumCircuit(qubits, clbits)
        expected.if_test(cond, empty, [], [clbits[0]])
        expected.if_else(cond, only_x, empty_with_qubit, [qubits[0]], [clbits[0]])
        expected.if_else(cond, empty_with_qubit, only_x, [qubits[0]], [clbits[0]])
        expected.if_else(cond, empty, empty, [], [clbits[0]])

        self.assertCircuitsEquivalent(test, expected)

    def test_if_else_tracks_registers(self):
        """Test that classical registers used in both branches of if statements are tracked
        correctly."""
        qr = QuantumRegister(2)
        cr = [ClassicalRegister(2) for _ in [None] * 4]

        test = QuantumCircuit(qr, *cr)
        with test.if_test((cr[0], 0)) as else_:
            test.h(0).c_if(cr[1], 0)
            # Test repetition.
            test.h(0).c_if(cr[1], 0)
        with else_:
            test.h(0).c_if(cr[2], 0)

        true_body = QuantumCircuit([qr[0]], cr[0], cr[1], cr[2])
        true_body.h(qr[0]).c_if(cr[1], 0)
        true_body.h(qr[0]).c_if(cr[1], 0)
        false_body = QuantumCircuit([qr[0]], cr[0], cr[1], cr[2])
        false_body.h(qr[0]).c_if(cr[2], 0)

        expected = QuantumCircuit(qr, *cr)
        expected.if_else(
            (cr[0], 0), true_body, false_body, [qr[0]], list(cr[0]) + list(cr[1]) + list(cr[2])
        )

        self.assertCircuitsEquivalent(test, expected)

    def test_if_else_nested(self):
        """Test that the if and else context managers can be nested, and don't interfere with each
        other."""
        qubits = [Qubit(), Qubit(), Qubit()]
        clbits = [Clbit(), Clbit(), Clbit()]

        outer_cond = (clbits[0], 0)
        inner_cond = (clbits[2], 1)

        with self.subTest("if (if) else"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(outer_cond) as else_:
                with test.if_test(inner_cond):
                    test.h(0)
            with else_:
                test.h(1)

            inner_true = QuantumCircuit([qubits[0], clbits[2]])
            inner_true.h(qubits[0])

            outer_true = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[2]])
            outer_true.if_test(inner_cond, inner_true, [qubits[0]], [clbits[2]])
            outer_false = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[2]])
            outer_false.h(qubits[1])

            expected = QuantumCircuit(qubits, clbits)
            expected.if_else(
                outer_cond, outer_true, outer_false, [qubits[0], qubits[1]], [clbits[0], clbits[2]]
            )
            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("if (if else) else"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(outer_cond) as outer_else:
                with test.if_test(inner_cond) as inner_else:
                    test.h(0)
                with inner_else:
                    test.h(2)
            with outer_else:
                test.h(1)

            inner_true = QuantumCircuit([qubits[0], qubits[2], clbits[2]])
            inner_true.h(qubits[0])
            inner_false = QuantumCircuit([qubits[0], qubits[2], clbits[2]])
            inner_false.h(qubits[2])

            outer_true = QuantumCircuit(qubits, [clbits[0], clbits[2]])
            outer_true.if_else(
                inner_cond, inner_true, inner_false, [qubits[0], qubits[2]], [clbits[2]]
            )
            outer_false = QuantumCircuit(qubits, [clbits[0], clbits[2]])
            outer_false.h(qubits[1])

            expected = QuantumCircuit(qubits, clbits)
            expected.if_else(outer_cond, outer_true, outer_false, qubits, [clbits[0], clbits[2]])
            self.assertCircuitsEquivalent(test, expected)

    def test_break_continue_expand_to_match_arguments_simple(self):
        """Test that ``break`` and ``continue`` statements expand to include all resources in the
        containing loop for simple cases with unconditional breaks."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]

        with self.subTest("for"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                test.break_loop()
                test.h(0)
                test.continue_loop()
                test.measure(1, 0)

            body = QuantumCircuit(qubits, [clbits[0]])
            body.break_loop()
            body.h(qubits[0])
            body.continue_loop()
            body.measure(qubits[1], clbits[0])

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(range(2), None, body, qubits, [clbits[0]])

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("while"):
            cond = (clbits[0], 0)
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond):
                test.break_loop()
                test.h(0)
                test.continue_loop()
                test.measure(1, 0)

            body = QuantumCircuit(qubits, [clbits[0]])
            body.break_loop()
            body.h(qubits[0])
            body.continue_loop()
            body.measure(qubits[1], clbits[0])

            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(cond, body, qubits, [clbits[0]])

            self.assertCircuitsEquivalent(test, expected)

    @ddt.data(QuantumCircuit.break_loop, QuantumCircuit.continue_loop)
    def test_break_continue_accept_c_if(self, loop_operation):
        """Test that ``break`` and ``continue`` statements accept :meth:`.Instruction.c_if` calls,
        and that these propagate through correctly."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]

        with self.subTest("for"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                test.h(0)
                loop_operation(test).c_if(1, 0)

            body = QuantumCircuit([qubits[0]], [clbits[1]])
            body.h(qubits[0])
            loop_operation(body).c_if(clbits[1], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(range(2), None, body, [qubits[0]], [clbits[1]])

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("while"):
            cond = (clbits[0], 0)
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond):
                test.h(0)
                loop_operation(test).c_if(1, 0)

            body = QuantumCircuit([qubits[0]], clbits)
            body.h(qubits[0])
            loop_operation(body).c_if(clbits[1], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(cond, body, [qubits[0]], clbits)

            self.assertCircuitsEquivalent(test, expected)

    @ddt.data(QuantumCircuit.break_loop, QuantumCircuit.continue_loop)
    def test_break_continue_only_expand_to_nearest_loop(self, loop_operation):
        """Test that a ``break`` or ``continue`` nested in more than one loop only expands as far as
        the inner loop scope, not further."""
        qubits = [Qubit(), Qubit(), Qubit()]
        clbits = [Clbit(), Clbit(), Clbit()]
        cond_inner = (clbits[0], 0)
        cond_outer = (clbits[1], 0)

        with self.subTest("for for"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                test.measure(1, 1)
                with test.for_loop(range(2)):
                    test.h(0)
                    loop_operation(test)
                loop_operation(test)

            inner_body = QuantumCircuit([qubits[0]])
            inner_body.h(qubits[0])
            loop_operation(inner_body)

            outer_body = QuantumCircuit([qubits[0], qubits[1], clbits[1]])
            outer_body.measure(qubits[1], clbits[1])
            outer_body.for_loop(range(2), None, inner_body, [qubits[0]], [])
            loop_operation(outer_body)

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(range(2), None, outer_body, [qubits[0], qubits[1]], [clbits[1]])

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("for while"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                test.measure(1, 1)
                with test.while_loop(cond_inner):
                    test.h(0)
                    loop_operation(test)
                loop_operation(test)

            inner_body = QuantumCircuit([qubits[0], clbits[0]])
            inner_body.h(qubits[0])
            loop_operation(inner_body)

            outer_body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1]])
            outer_body.measure(qubits[1], clbits[1])
            outer_body.while_loop(cond_inner, inner_body, [qubits[0]], [clbits[0]])
            loop_operation(outer_body)

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(
                range(2), None, outer_body, [qubits[0], qubits[1]], [clbits[0], clbits[1]]
            )

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("while for"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond_outer):
                test.measure(1, 1)
                with test.for_loop(range(2)):
                    test.h(0)
                    loop_operation(test)
                loop_operation(test)

            inner_body = QuantumCircuit([qubits[0]])
            inner_body.h(qubits[0])
            loop_operation(inner_body)

            outer_body = QuantumCircuit([qubits[0], qubits[1], clbits[1]])
            outer_body.measure(qubits[1], clbits[1])
            outer_body.for_loop(range(2), None, inner_body, [qubits[0]], [])
            loop_operation(outer_body)

            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(cond_outer, outer_body, [qubits[0], qubits[1]], [clbits[1]])

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("while while"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond_outer):
                test.measure(1, 1)
                with test.while_loop(cond_inner):
                    test.h(0)
                    loop_operation(test)
                loop_operation(test)

            inner_body = QuantumCircuit([qubits[0], clbits[0]])
            inner_body.h(qubits[0])
            loop_operation(inner_body)

            outer_body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1]])
            outer_body.measure(qubits[1], clbits[1])
            outer_body.while_loop(cond_inner, inner_body, [qubits[0]], [clbits[0]])
            loop_operation(outer_body)

            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(
                cond_outer, outer_body, [qubits[0], qubits[1]], [clbits[0], clbits[1]]
            )

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("for (for, for)"):
            # This test is specifically to check that multiple inner loops with different numbers of
            # variables to each ``break`` still expand correctly.
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                test.measure(1, 1)
                with test.for_loop(range(2)):
                    test.h(0)
                    loop_operation(test)
                with test.for_loop(range(2)):
                    test.h(2)
                    loop_operation(test)
                loop_operation(test)

            inner_body1 = QuantumCircuit([qubits[0]])
            inner_body1.h(qubits[0])
            loop_operation(inner_body1)

            inner_body2 = QuantumCircuit([qubits[2]])
            inner_body2.h(qubits[2])
            loop_operation(inner_body2)

            outer_body = QuantumCircuit([qubits[0], qubits[1], qubits[2], clbits[1]])
            outer_body.measure(qubits[1], clbits[1])
            outer_body.for_loop(range(2), None, inner_body1, [qubits[0]], [])
            outer_body.for_loop(range(2), None, inner_body2, [qubits[2]], [])
            loop_operation(outer_body)

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(range(2), None, outer_body, qubits, [clbits[1]])

            self.assertCircuitsEquivalent(test, expected)

    @ddt.data(QuantumCircuit.break_loop, QuantumCircuit.continue_loop)
    def test_break_continue_nested_in_if(self, loop_operation):
        """Test that ``break`` and ``continue`` work correctly when inside an ``if`` block within a
        loop.  This includes testing that multiple different ``if`` statements with and without
        ``break`` expand to the correct number of arguments.

        This is a very important case; it requires that the :obj:`.IfElseOp` is not built until the
        loop builds, and that the width expands to include everything that the loop knows about, not
        just the inner context.  We test both ``if`` and ``if/else`` paths separately, because the
        chaining of the context managers allows lots of possibilities for super weird edge cases.

        There are several tests that build up in complexity to aid debugging if something goes
        wrong; the aim is that there will be more information available depending on which of the
        subtests pass and which fail.
        """
        qubits = [Qubit(), Qubit(), Qubit()]
        clbits = [Clbit(), Clbit(), Clbit(), Clbit()]
        cond_inner = (clbits[0], 0)
        cond_outer = (clbits[1], 0)

        with self.subTest("for/if"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                with test.if_test(cond_inner):
                    loop_operation(test)
                # The second empty `if` is to test that only blocks that _need_ to expand to be the
                # full width of the loop do so.
                with test.if_test(cond_inner):
                    pass
                test.h(0).c_if(2, 0)

            true_body1 = QuantumCircuit([qubits[0], clbits[0], clbits[2]])
            loop_operation(true_body1)

            true_body2 = QuantumCircuit([clbits[0]])

            loop_body = QuantumCircuit([qubits[0], clbits[0], clbits[2]])
            loop_body.if_test(cond_inner, true_body1, [qubits[0]], [clbits[0], clbits[2]])
            loop_body.if_test(cond_inner, true_body2, [], [clbits[0]])
            loop_body.h(qubits[0]).c_if(clbits[2], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(range(2), None, loop_body, [qubits[0]], [clbits[0], clbits[2]])

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("for/else"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                with test.if_test(cond_inner) as else_:
                    test.h(1)
                with else_:
                    loop_operation(test)
                with test.if_test(cond_inner) as else_:
                    pass
                with else_:
                    pass
                test.h(0).c_if(2, 0)

            true_body1 = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[2]])
            true_body1.h(qubits[1])
            false_body1 = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[2]])
            loop_operation(false_body1)

            true_body2 = QuantumCircuit([clbits[0]])
            false_body2 = QuantumCircuit([clbits[0]])

            loop_body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[2]])
            loop_body.if_else(
                cond_inner, true_body1, false_body1, [qubits[0], qubits[1]], [clbits[0], clbits[2]]
            )
            loop_body.if_else(cond_inner, true_body2, false_body2, [], [clbits[0]])
            loop_body.h(qubits[0]).c_if(clbits[2], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(
                range(2), None, loop_body, [qubits[0], qubits[1]], [clbits[0], clbits[2]]
            )

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("while/if"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond_outer):
                with test.if_test(cond_inner):
                    loop_operation(test)
                with test.if_test(cond_inner):
                    pass
                test.h(0).c_if(2, 0)

            true_body1 = QuantumCircuit([qubits[0], clbits[0], clbits[1], clbits[2]])
            loop_operation(true_body1)

            true_body2 = QuantumCircuit([clbits[0]])

            loop_body = QuantumCircuit([qubits[0], clbits[0], clbits[1], clbits[2]])
            loop_body.if_test(
                cond_inner, true_body1, [qubits[0]], [clbits[0], clbits[1], clbits[2]]
            )
            loop_body.if_test(cond_inner, true_body2, [], [clbits[0]])
            loop_body.h(qubits[0]).c_if(clbits[2], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(
                cond_outer, loop_body, [qubits[0]], [clbits[0], clbits[1], clbits[2]]
            )

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("while/else"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond_outer):
                with test.if_test(cond_inner) as else_:
                    test.h(1)
                with else_:
                    loop_operation(test)
                with test.if_test(cond_inner) as else_:
                    pass
                with else_:
                    pass
                test.h(0).c_if(2, 0)

            true_body1 = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1], clbits[2]])
            true_body1.h(qubits[1])
            false_body1 = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1], clbits[2]])
            loop_operation(false_body1)

            true_body2 = QuantumCircuit([clbits[0]])
            false_body2 = QuantumCircuit([clbits[0]])

            loop_body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1], clbits[2]])
            loop_body.if_else(
                cond_inner,
                true_body1,
                false_body1,
                [qubits[0], qubits[1]],
                [clbits[0], clbits[1], clbits[2]],
            )
            loop_body.if_else(cond_inner, true_body2, false_body2, [], [clbits[0]])
            loop_body.h(qubits[0]).c_if(clbits[2], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(
                cond_outer, loop_body, [qubits[0], qubits[1]], [clbits[0], clbits[1], clbits[2]]
            )

            self.assertCircuitsEquivalent(test, expected)

    @ddt.data(QuantumCircuit.break_loop, QuantumCircuit.continue_loop)
    def test_break_continue_deeply_nested_in_if(self, loop_operation):
        """Test that ``break`` and ``continue`` work correctly when inside more than one ``if``
        block within a loop.  This includes testing that multiple different ``if`` statements with
        and without ``break`` expand to the correct number of arguments.

        These are the deepest tests, hitting all parts of the deferred builder scopes.  We test both
        ``if`` and ``if/else`` paths at various levels of the scoping to try and account for as many
        weird edge cases with the deferred behaviour as possible.  We try to make sure, particularly
        in the most complicated examples, that there are resources added before and after every
        single scope, to try and catch all possibilities of where resources may be missed.

        There are several tests that build up in complexity to aid debugging if something goes
        wrong; the aim is that there will be more information available depending on which of the
        subtests pass and which fail.
        """
        # These are deliberately more than is absolutely needed so we can detect if extra resources
        # are being erroneously included as well.
        qubits = [Qubit() for _ in [None] * 20]
        clbits = [Clbit() for _ in [None] * 20]
        cond_inner = (clbits[0], 0)
        cond_outer = (clbits[1], 0)
        cond_loop = (clbits[2], 0)

        with self.subTest("for/if/if"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                # outer true 1
                with test.if_test(cond_outer):
                    # inner true 1
                    with test.if_test(cond_inner):
                        loop_operation(test)
                    # inner true 2
                    with test.if_test(cond_inner):
                        test.h(0).c_if(3, 0)
                    test.h(1).c_if(4, 0)
                # outer true 2
                with test.if_test(cond_outer):
                    test.h(2).c_if(5, 0)
                test.h(3).c_if(6, 0)

            inner_true_body1 = QuantumCircuit(qubits[:4], clbits[:2], clbits[3:7])
            loop_operation(inner_true_body1)

            inner_true_body2 = QuantumCircuit([qubits[0], clbits[0], clbits[3]])
            inner_true_body2.h(qubits[0]).c_if(clbits[3], 0)

            outer_true_body1 = QuantumCircuit(qubits[:4], clbits[:2], clbits[3:7])
            outer_true_body1.if_test(
                cond_inner, inner_true_body1, qubits[:4], clbits[:2] + clbits[3:7]
            )
            outer_true_body1.if_test(
                cond_inner, inner_true_body2, [qubits[0]], [clbits[0], clbits[3]]
            )
            outer_true_body1.h(qubits[1]).c_if(clbits[4], 0)

            outer_true_body2 = QuantumCircuit([qubits[2], clbits[1], clbits[5]])
            outer_true_body2.h(qubits[2]).c_if(clbits[5], 0)

            loop_body = QuantumCircuit(qubits[:4], clbits[:2] + clbits[3:7])
            loop_body.if_test(cond_outer, outer_true_body1, qubits[:4], clbits[:2] + clbits[3:7])
            loop_body.if_test(cond_outer, outer_true_body2, [qubits[2]], [clbits[1], clbits[5]])
            loop_body.h(qubits[3]).c_if(clbits[6], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(range(2), None, loop_body, qubits[:4], clbits[:2] + clbits[3:7])

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("for/if/else"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                # outer 1
                with test.if_test(cond_outer):
                    # inner 1
                    with test.if_test(cond_inner) as inner1_else:
                        test.h(0).c_if(3, 0)
                    with inner1_else:
                        loop_operation(test).c_if(4, 0)
                    # inner 2
                    with test.if_test(cond_inner) as inner2_else:
                        test.h(1).c_if(5, 0)
                    with inner2_else:
                        test.h(2).c_if(6, 0)
                    test.h(3).c_if(7, 0)
                # outer 2
                with test.if_test(cond_outer) as outer2_else:
                    test.h(4).c_if(8, 0)
                with outer2_else:
                    test.h(5).c_if(9, 0)
                test.h(6).c_if(10, 0)

            inner1_true = QuantumCircuit(qubits[:7], clbits[:2], clbits[3:11])
            inner1_true.h(qubits[0]).c_if(clbits[3], 0)
            inner1_false = QuantumCircuit(qubits[:7], clbits[:2], clbits[3:11])
            loop_operation(inner1_false).c_if(clbits[4], 0)

            inner2_true = QuantumCircuit([qubits[1], qubits[2], clbits[0], clbits[5], clbits[6]])
            inner2_true.h(qubits[1]).c_if(clbits[5], 0)
            inner2_false = QuantumCircuit([qubits[1], qubits[2], clbits[0], clbits[5], clbits[6]])
            inner2_false.h(qubits[2]).c_if(clbits[6], 0)

            outer1_true = QuantumCircuit(qubits[:7], clbits[:2], clbits[3:11])
            outer1_true.if_else(
                cond_inner, inner1_true, inner1_false, qubits[:7], clbits[:2] + clbits[3:11]
            )
            outer1_true.if_else(
                cond_inner,
                inner2_true,
                inner2_false,
                qubits[1:3],
                [clbits[0], clbits[5], clbits[6]],
            )
            outer1_true.h(qubits[3]).c_if(clbits[7], 0)

            outer2_true = QuantumCircuit([qubits[4], qubits[5], clbits[1], clbits[8], clbits[9]])
            outer2_true.h(qubits[4]).c_if(clbits[8], 0)
            outer2_false = QuantumCircuit([qubits[4], qubits[5], clbits[1], clbits[8], clbits[9]])
            outer2_false.h(qubits[5]).c_if(clbits[9], 0)

            loop_body = QuantumCircuit(qubits[:7], clbits[:2], clbits[3:11])
            loop_body.if_test(cond_outer, outer1_true, qubits[:7], clbits[:2] + clbits[3:11])
            loop_body.if_else(
                cond_outer,
                outer2_true,
                outer2_false,
                qubits[4:6],
                [clbits[1], clbits[8], clbits[9]],
            )
            loop_body.h(qubits[6]).c_if(clbits[10], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(range(2), None, loop_body, qubits[:7], clbits[:2] + clbits[3:11])

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("for/else/else"):
            # Look on my works, ye Mighty, and despair!

            # (but also hopefully this is less hubristic pretension and more a useful stress test)

            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                test.h(0).c_if(3, 0)

                # outer 1
                with test.if_test(cond_outer) as outer1_else:
                    test.h(1).c_if(4, 0)
                with outer1_else:
                    test.h(2).c_if(5, 0)

                # outer 2 (nesting the inner condition in the 'if')
                with test.if_test(cond_outer) as outer2_else:
                    test.h(3).c_if(6, 0)

                    # inner 21
                    with test.if_test(cond_inner) as inner21_else:
                        loop_operation(test)
                    with inner21_else:
                        test.h(4).c_if(7, 0)

                    # inner 22
                    with test.if_test(cond_inner) as inner22_else:
                        test.h(5).c_if(8, 0)
                    with inner22_else:
                        loop_operation(test)

                    test.h(6).c_if(9, 0)
                with outer2_else:
                    test.h(7).c_if(10, 0)

                    # inner 23
                    with test.if_test(cond_inner) as inner23_else:
                        test.h(8).c_if(11, 0)
                    with inner23_else:
                        test.h(9).c_if(12, 0)

                # outer 3 (nesting the inner condition in an 'else' branch)
                with test.if_test(cond_outer) as outer3_else:
                    test.h(10).c_if(13, 0)
                with outer3_else:
                    test.h(11).c_if(14, 0)

                    # inner 31
                    with test.if_test(cond_inner) as inner31_else:
                        loop_operation(test)
                    with inner31_else:
                        test.h(12).c_if(15, 0)

                    # inner 32
                    with test.if_test(cond_inner) as inner32_else:
                        test.h(13).c_if(16, 0)
                    with inner32_else:
                        loop_operation(test)

                    # inner 33
                    with test.if_test(cond_inner) as inner33_else:
                        test.h(14).c_if(17, 0)
                    with inner33_else:
                        test.h(15).c_if(18, 0)

                test.h(16).c_if(19, 0)
            # End of test "for" loop.

            # No `clbits[2]` here because that's only used in `cond_loop`, for while loops.
            loop_qubits = qubits[:17]
            loop_clbits = clbits[:2] + clbits[3:20]
            loop_bits = loop_qubits + loop_clbits

            outer1_true = QuantumCircuit([qubits[1], qubits[2], clbits[1], clbits[4], clbits[5]])
            outer1_true.h(qubits[1]).c_if(clbits[4], 0)
            outer1_false = QuantumCircuit([qubits[1], qubits[2], clbits[1], clbits[4], clbits[5]])
            outer1_false.h(qubits[2]).c_if(clbits[5], 0)

            inner21_true = QuantumCircuit(loop_bits)
            loop_operation(inner21_true)
            inner21_false = QuantumCircuit(loop_bits)
            inner21_false.h(qubits[4]).c_if(clbits[7], 0)

            inner22_true = QuantumCircuit(loop_bits)
            inner22_true.h(qubits[5]).c_if(clbits[8], 0)
            inner22_false = QuantumCircuit(loop_bits)
            loop_operation(inner22_false)

            inner23_true = QuantumCircuit(qubits[8:10], [clbits[0], clbits[11], clbits[12]])
            inner23_true.h(qubits[8]).c_if(clbits[11], 0)
            inner23_false = QuantumCircuit(qubits[8:10], [clbits[0], clbits[11], clbits[12]])
            inner23_false.h(qubits[9]).c_if(clbits[12], 0)

            outer2_true = QuantumCircuit(loop_bits)
            outer2_true.h(qubits[3]).c_if(clbits[6], 0)
            outer2_true.if_else(cond_inner, inner21_true, inner21_false, loop_qubits, loop_clbits)
            outer2_true.if_else(cond_inner, inner22_true, inner22_false, loop_qubits, loop_clbits)
            outer2_true.h(qubits[6]).c_if(clbits[9], 0)
            outer2_false = QuantumCircuit(loop_bits)
            outer2_false.h(qubits[7]).c_if(clbits[10], 0)
            outer2_false.if_else(
                cond_inner,
                inner23_true,
                inner23_false,
                [qubits[8], qubits[9]],
                [clbits[0], clbits[11], clbits[12]],
            )

            inner31_true = QuantumCircuit(loop_bits)
            loop_operation(inner31_true)
            inner31_false = QuantumCircuit(loop_bits)
            inner31_false.h(qubits[12]).c_if(clbits[15], 0)

            inner32_true = QuantumCircuit(loop_bits)
            inner32_true.h(qubits[13]).c_if(clbits[16], 0)
            inner32_false = QuantumCircuit(loop_bits)
            loop_operation(inner32_false)

            inner33_true = QuantumCircuit(qubits[14:16], [clbits[0], clbits[17], clbits[18]])
            inner33_true.h(qubits[14]).c_if(clbits[17], 0)
            inner33_false = QuantumCircuit(qubits[14:16], [clbits[0], clbits[17], clbits[18]])
            inner33_false.h(qubits[15]).c_if(clbits[18], 0)

            outer3_true = QuantumCircuit(loop_bits)
            outer3_true.h(qubits[10]).c_if(clbits[13], 0)
            outer3_false = QuantumCircuit(loop_bits)
            outer3_false.h(qubits[11]).c_if(clbits[14], 0)
            outer3_false.if_else(cond_inner, inner31_true, inner31_false, loop_qubits, loop_clbits)
            outer3_false.if_else(cond_inner, inner32_true, inner32_false, loop_qubits, loop_clbits)
            outer3_false.if_else(
                cond_inner,
                inner33_true,
                inner33_false,
                qubits[14:16],
                [clbits[0], clbits[17], clbits[18]],
            )

            loop_body = QuantumCircuit(loop_bits)
            loop_body.h(qubits[0]).c_if(clbits[3], 0)
            loop_body.if_else(
                cond_outer,
                outer1_true,
                outer1_false,
                qubits[1:3],
                [clbits[1], clbits[4], clbits[5]],
            )
            loop_body.if_else(cond_outer, outer2_true, outer2_false, loop_qubits, loop_clbits)
            loop_body.if_else(cond_outer, outer3_true, outer3_false, loop_qubits, loop_clbits)
            loop_body.h(qubits[16]).c_if(clbits[19], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(range(2), None, loop_body, loop_qubits, loop_clbits)

            self.assertCircuitsEquivalent(test, expected)

        # And now we repeat everything for "while" loops...  Trying to parameterize the test over
        # 'for/while' mostly just ends up in it being a bit illegible, because so many parameters
        # vary in the explicit construction form.  These tests are just copies of the above tests,
        # but with `while_loop(cond_loop)` instead of `for_loop(range(2))`, and the corresponding
        # clbit ranges updated to include `clbits[2]` from the condition.

        with self.subTest("while/if/if"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond_loop):
                # outer true 1
                with test.if_test(cond_outer):
                    # inner true 1
                    with test.if_test(cond_inner):
                        loop_operation(test)
                    # inner true 2
                    with test.if_test(cond_inner):
                        test.h(0).c_if(3, 0)
                    test.h(1).c_if(4, 0)
                # outer true 2
                with test.if_test(cond_outer):
                    test.h(2).c_if(5, 0)
                test.h(3).c_if(6, 0)

            inner_true_body1 = QuantumCircuit(qubits[:4], clbits[:7])
            loop_operation(inner_true_body1)

            inner_true_body2 = QuantumCircuit([qubits[0], clbits[0], clbits[3]])
            inner_true_body2.h(qubits[0]).c_if(clbits[3], 0)

            outer_true_body1 = QuantumCircuit(qubits[:4], clbits[:7])
            outer_true_body1.if_test(cond_inner, inner_true_body1, qubits[:4], clbits[:7])
            outer_true_body1.if_test(
                cond_inner, inner_true_body2, [qubits[0]], [clbits[0], clbits[3]]
            )
            outer_true_body1.h(qubits[1]).c_if(clbits[4], 0)

            outer_true_body2 = QuantumCircuit([qubits[2], clbits[1], clbits[5]])
            outer_true_body2.h(qubits[2]).c_if(clbits[5], 0)

            loop_body = QuantumCircuit(qubits[:4], clbits[:7])
            loop_body.if_test(cond_outer, outer_true_body1, qubits[:4], clbits[:7])
            loop_body.if_test(cond_outer, outer_true_body2, [qubits[2]], [clbits[1], clbits[5]])
            loop_body.h(qubits[3]).c_if(clbits[6], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(cond_loop, loop_body, qubits[:4], clbits[:7])

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("while/if/else"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond_loop):
                # outer 1
                with test.if_test(cond_outer):
                    # inner 1
                    with test.if_test(cond_inner) as inner1_else:
                        test.h(0).c_if(3, 0)
                    with inner1_else:
                        loop_operation(test).c_if(4, 0)
                    # inner 2
                    with test.if_test(cond_inner) as inner2_else:
                        test.h(1).c_if(5, 0)
                    with inner2_else:
                        test.h(2).c_if(6, 0)
                    test.h(3).c_if(7, 0)
                # outer 2
                with test.if_test(cond_outer) as outer2_else:
                    test.h(4).c_if(8, 0)
                with outer2_else:
                    test.h(5).c_if(9, 0)
                test.h(6).c_if(10, 0)

            inner1_true = QuantumCircuit(qubits[:7], clbits[:11])
            inner1_true.h(qubits[0]).c_if(clbits[3], 0)
            inner1_false = QuantumCircuit(qubits[:7], clbits[:11])
            loop_operation(inner1_false).c_if(clbits[4], 0)

            inner2_true = QuantumCircuit([qubits[1], qubits[2], clbits[0], clbits[5], clbits[6]])
            inner2_true.h(qubits[1]).c_if(clbits[5], 0)
            inner2_false = QuantumCircuit([qubits[1], qubits[2], clbits[0], clbits[5], clbits[6]])
            inner2_false.h(qubits[2]).c_if(clbits[6], 0)

            outer1_true = QuantumCircuit(qubits[:7], clbits[:11])
            outer1_true.if_else(cond_inner, inner1_true, inner1_false, qubits[:7], clbits[:11])
            outer1_true.if_else(
                cond_inner,
                inner2_true,
                inner2_false,
                qubits[1:3],
                [clbits[0], clbits[5], clbits[6]],
            )
            outer1_true.h(qubits[3]).c_if(clbits[7], 0)

            outer2_true = QuantumCircuit([qubits[4], qubits[5], clbits[1], clbits[8], clbits[9]])
            outer2_true.h(qubits[4]).c_if(clbits[8], 0)
            outer2_false = QuantumCircuit([qubits[4], qubits[5], clbits[1], clbits[8], clbits[9]])
            outer2_false.h(qubits[5]).c_if(clbits[9], 0)

            loop_body = QuantumCircuit(qubits[:7], clbits[:11])
            loop_body.if_test(cond_outer, outer1_true, qubits[:7], clbits[:11])
            loop_body.if_else(
                cond_outer,
                outer2_true,
                outer2_false,
                qubits[4:6],
                [clbits[1], clbits[8], clbits[9]],
            )
            loop_body.h(qubits[6]).c_if(clbits[10], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(cond_loop, loop_body, qubits[:7], clbits[:11])

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("while/else/else"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond_loop):
                test.h(0).c_if(3, 0)

                # outer 1
                with test.if_test(cond_outer) as outer1_else:
                    test.h(1).c_if(4, 0)
                with outer1_else:
                    test.h(2).c_if(5, 0)

                # outer 2 (nesting the inner condition in the 'if')
                with test.if_test(cond_outer) as outer2_else:
                    test.h(3).c_if(6, 0)

                    # inner 21
                    with test.if_test(cond_inner) as inner21_else:
                        loop_operation(test)
                    with inner21_else:
                        test.h(4).c_if(7, 0)

                    # inner 22
                    with test.if_test(cond_inner) as inner22_else:
                        test.h(5).c_if(8, 0)
                    with inner22_else:
                        loop_operation(test)

                    test.h(6).c_if(9, 0)
                with outer2_else:
                    test.h(7).c_if(10, 0)

                    # inner 23
                    with test.if_test(cond_inner) as inner23_else:
                        test.h(8).c_if(11, 0)
                    with inner23_else:
                        test.h(9).c_if(12, 0)

                # outer 3 (nesting the inner condition in an 'else' branch)
                with test.if_test(cond_outer) as outer3_else:
                    test.h(10).c_if(13, 0)
                with outer3_else:
                    test.h(11).c_if(14, 0)

                    # inner 31
                    with test.if_test(cond_inner) as inner31_else:
                        loop_operation(test)
                    with inner31_else:
                        test.h(12).c_if(15, 0)

                    # inner 32
                    with test.if_test(cond_inner) as inner32_else:
                        test.h(13).c_if(16, 0)
                    with inner32_else:
                        loop_operation(test)

                    # inner 33
                    with test.if_test(cond_inner) as inner33_else:
                        test.h(14).c_if(17, 0)
                    with inner33_else:
                        test.h(15).c_if(18, 0)

                test.h(16).c_if(19, 0)
            # End of test "for" loop.

            # No `clbits[2]` here because that's only used in `cond_loop`, for while loops.
            loop_qubits = qubits[:17]
            loop_clbits = clbits[:20]
            loop_bits = loop_qubits + loop_clbits

            outer1_true = QuantumCircuit([qubits[1], qubits[2], clbits[1], clbits[4], clbits[5]])
            outer1_true.h(qubits[1]).c_if(clbits[4], 0)
            outer1_false = QuantumCircuit([qubits[1], qubits[2], clbits[1], clbits[4], clbits[5]])
            outer1_false.h(qubits[2]).c_if(clbits[5], 0)

            inner21_true = QuantumCircuit(loop_bits)
            loop_operation(inner21_true)
            inner21_false = QuantumCircuit(loop_bits)
            inner21_false.h(qubits[4]).c_if(clbits[7], 0)

            inner22_true = QuantumCircuit(loop_bits)
            inner22_true.h(qubits[5]).c_if(clbits[8], 0)
            inner22_false = QuantumCircuit(loop_bits)
            loop_operation(inner22_false)

            inner23_true = QuantumCircuit(qubits[8:10], [clbits[0], clbits[11], clbits[12]])
            inner23_true.h(qubits[8]).c_if(clbits[11], 0)
            inner23_false = QuantumCircuit(qubits[8:10], [clbits[0], clbits[11], clbits[12]])
            inner23_false.h(qubits[9]).c_if(clbits[12], 0)

            outer2_true = QuantumCircuit(loop_bits)
            outer2_true.h(qubits[3]).c_if(clbits[6], 0)
            outer2_true.if_else(cond_inner, inner21_true, inner21_false, loop_qubits, loop_clbits)
            outer2_true.if_else(cond_inner, inner22_true, inner22_false, loop_qubits, loop_clbits)
            outer2_true.h(qubits[6]).c_if(clbits[9], 0)
            outer2_false = QuantumCircuit(loop_bits)
            outer2_false.h(qubits[7]).c_if(clbits[10], 0)
            outer2_false.if_else(
                cond_inner,
                inner23_true,
                inner23_false,
                [qubits[8], qubits[9]],
                [clbits[0], clbits[11], clbits[12]],
            )

            inner31_true = QuantumCircuit(loop_bits)
            loop_operation(inner31_true)
            inner31_false = QuantumCircuit(loop_bits)
            inner31_false.h(qubits[12]).c_if(clbits[15], 0)

            inner32_true = QuantumCircuit(loop_bits)
            inner32_true.h(qubits[13]).c_if(clbits[16], 0)
            inner32_false = QuantumCircuit(loop_bits)
            loop_operation(inner32_false)

            inner33_true = QuantumCircuit(qubits[14:16], [clbits[0], clbits[17], clbits[18]])
            inner33_true.h(qubits[14]).c_if(clbits[17], 0)
            inner33_false = QuantumCircuit(qubits[14:16], [clbits[0], clbits[17], clbits[18]])
            inner33_false.h(qubits[15]).c_if(clbits[18], 0)

            outer3_true = QuantumCircuit(loop_bits)
            outer3_true.h(qubits[10]).c_if(clbits[13], 0)
            outer3_false = QuantumCircuit(loop_bits)
            outer3_false.h(qubits[11]).c_if(clbits[14], 0)
            outer3_false.if_else(cond_inner, inner31_true, inner31_false, loop_qubits, loop_clbits)
            outer3_false.if_else(cond_inner, inner32_true, inner32_false, loop_qubits, loop_clbits)
            outer3_false.if_else(
                cond_inner,
                inner33_true,
                inner33_false,
                qubits[14:16],
                [clbits[0], clbits[17], clbits[18]],
            )

            loop_body = QuantumCircuit(loop_bits)
            loop_body.h(qubits[0]).c_if(clbits[3], 0)
            loop_body.if_else(
                cond_outer,
                outer1_true,
                outer1_false,
                qubits[1:3],
                [clbits[1], clbits[4], clbits[5]],
            )
            loop_body.if_else(cond_outer, outer2_true, outer2_false, loop_qubits, loop_clbits)
            loop_body.if_else(cond_outer, outer3_true, outer3_false, loop_qubits, loop_clbits)
            loop_body.h(qubits[16]).c_if(clbits[19], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(cond_loop, loop_body, loop_qubits, loop_clbits)

            self.assertCircuitsEquivalent(test, expected)

    def test_for_handles_iterables_correctly(self):
        """Test that the ``indexset`` in ``for`` loops is handled the way we expect.  In general,
        this means all iterables are consumed into a tuple on first access, except for ``range``
        which is passed through as-is."""
        bits = [Qubit(), Clbit()]
        expected_indices = (3, 9, 1)

        with self.subTest("list"):
            test = QuantumCircuit(bits)
            with test.for_loop(list(expected_indices)):
                pass
            instruction, _, _ = test.data[-1]
            self.assertIsInstance(instruction, ForLoopOp)
            indices, _, _ = instruction.params
            self.assertEqual(indices, expected_indices)

        with self.subTest("tuple"):
            test = QuantumCircuit(bits)
            with test.for_loop(tuple(expected_indices)):
                pass
            instruction, _, _ = test.data[-1]
            self.assertIsInstance(instruction, ForLoopOp)
            indices, _, _ = instruction.params
            self.assertEqual(indices, expected_indices)

        with self.subTest("consumable"):

            def consumable():
                yield from expected_indices

            test = QuantumCircuit(bits)
            with test.for_loop(consumable()):
                pass
            instruction, _, _ = test.data[-1]
            self.assertIsInstance(instruction, ForLoopOp)
            indices, _, _ = instruction.params
            self.assertEqual(indices, expected_indices)

        with self.subTest("range"):
            range_indices = range(0, 8, 2)

            test = QuantumCircuit(bits)
            with test.for_loop(range_indices):
                pass
            instruction, _, _ = test.data[-1]
            self.assertIsInstance(instruction, ForLoopOp)
            indices, _, _ = instruction.params
            self.assertEqual(indices, range_indices)

    def test_for_returns_a_given_parameter(self):
        """Test that the ``for``-loop manager returns the parameter that we gave it."""
        parameter = Parameter("x")
        test = QuantumCircuit(1, 1)
        with test.for_loop((0, 1), parameter) as test_parameter:
            pass
        self.assertIs(test_parameter, parameter)

    def test_for_binds_parameter_to_op(self):
        """Test that the ``for`` manager binds a parameter to the resulting :obj:`.ForLoopOp` if a
        user-generated one is given, or if a generated parameter is used.  Generated parameters that
        are not used should not be bound."""
        parameter = Parameter("x")

        with self.subTest("passed and used"):
            circuit = QuantumCircuit(1, 1)
            with circuit.for_loop((0, 0.5 * math.pi), parameter) as received_parameter:
                circuit.rx(received_parameter, 0)
            self.assertIs(parameter, received_parameter)
            instruction = circuit.data[-1][0]
            self.assertIsInstance(instruction, ForLoopOp)
            _, bound_parameter, _ = instruction.params
            self.assertIs(bound_parameter, parameter)

        with self.subTest("passed and unused"):
            circuit = QuantumCircuit(1, 1)
            with circuit.for_loop((0, 0.5 * math.pi), parameter) as received_parameter:
                circuit.x(0)
            self.assertIs(parameter, received_parameter)
            instruction = circuit.data[-1][0]
            self.assertIsInstance(instruction, ForLoopOp)
            _, bound_parameter, _ = instruction.params
            self.assertIs(parameter, received_parameter)

        with self.subTest("generated and used"):
            circuit = QuantumCircuit(1, 1)
            with circuit.for_loop((0, 0.5 * math.pi)) as received_parameter:
                circuit.rx(received_parameter, 0)
            self.assertIsInstance(received_parameter, Parameter)
            instruction = circuit.data[-1][0]
            self.assertIsInstance(instruction, ForLoopOp)
            _, bound_parameter, _ = instruction.params
            self.assertIs(bound_parameter, received_parameter)

        with self.subTest("generated and used in deferred-build if"):
            circuit = QuantumCircuit(1, 1)
            with circuit.for_loop((0, 0.5 * math.pi)) as received_parameter:
                with circuit.if_test((0, 0)):
                    circuit.rx(received_parameter, 0)
                    circuit.break_loop()
            self.assertIsInstance(received_parameter, Parameter)
            instruction = circuit.data[-1][0]
            self.assertIsInstance(instruction, ForLoopOp)
            _, bound_parameter, _ = instruction.params
            self.assertIs(bound_parameter, received_parameter)

        with self.subTest("generated and used in deferred-build else"):
            circuit = QuantumCircuit(1, 1)
            with circuit.for_loop((0, 0.5 * math.pi)) as received_parameter:
                with circuit.if_test((0, 0)) as else_:
                    pass
                with else_:
                    circuit.rx(received_parameter, 0)
                    circuit.break_loop()
            self.assertIsInstance(received_parameter, Parameter)
            instruction = circuit.data[-1][0]
            self.assertIsInstance(instruction, ForLoopOp)
            _, bound_parameter, _ = instruction.params
            self.assertIs(bound_parameter, received_parameter)

    def test_for_does_not_bind_generated_parameter_if_unused(self):
        """Test that the ``for`` manager does not bind a generated parameter into the resulting
        :obj:`.ForLoopOp` if the parameter was not used."""
        test = QuantumCircuit(1, 1)
        with test.for_loop(range(2)) as generated_parameter:
            pass
        instruction = test.data[-1][0]
        self.assertIsInstance(instruction, ForLoopOp)
        _, bound_parameter, _ = instruction.params
        self.assertIsNot(generated_parameter, None)
        self.assertIs(bound_parameter, None)

    def test_for_allocates_parameters(self):
        """Test that the ``for``-loop manager allocates a parameter if it is given ``None``, and
        that it always allocates new parameters."""
        test = QuantumCircuit(1, 1)
        with test.for_loop(range(2)) as outer_parameter:
            with test.for_loop(range(2)) as inner_parameter:
                pass
        with test.for_loop(range(2)) as final_parameter:
            pass
        self.assertIsInstance(outer_parameter, Parameter)
        self.assertIsInstance(inner_parameter, Parameter)
        self.assertIsInstance(final_parameter, Parameter)
        self.assertNotEqual(outer_parameter, inner_parameter)
        self.assertNotEqual(outer_parameter, final_parameter)
        self.assertNotEqual(inner_parameter, final_parameter)

    def test_access_of_resources_from_direct_append(self):
        """Test that direct calls to :obj:`.QuantumCircuit.append` within a builder block still
        collect all the relevant resources."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]
        cond = (clbits[0], 0)

        with self.subTest("if"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond):
                test.append(Measure(), [qubits[1]], [clbits[1]])

            true_body = QuantumCircuit([qubits[1]], clbits)
            true_body.measure(qubits[1], clbits[1])
            expected = QuantumCircuit(qubits, clbits)
            expected.if_test(cond, true_body, [qubits[1]], clbits)

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("else"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond) as else_:
                pass
            with else_:
                test.append(Measure(), [qubits[1]], [clbits[1]])

            true_body = QuantumCircuit([qubits[1]], clbits)
            false_body = QuantumCircuit([qubits[1]], clbits)
            false_body.measure(qubits[1], clbits[1])
            expected = QuantumCircuit(qubits, clbits)
            expected.if_else(cond, true_body, false_body, [qubits[1]], clbits)

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("for"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                test.append(Measure(), [qubits[1]], [clbits[1]])

            body = QuantumCircuit([qubits[1]], [clbits[1]])
            body.measure(qubits[1], clbits[1])
            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(range(2), None, body, [qubits[1]], [clbits[1]])

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("while"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond):
                test.append(Measure(), [qubits[1]], [clbits[1]])

            body = QuantumCircuit([qubits[1]], clbits)
            body.measure(qubits[1], clbits[1])
            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(cond, body, [qubits[1]], clbits)

            self.assertCircuitsEquivalent(test, expected)

    def test_access_of_clbit_from_c_if(self):
        """Test that resources added from a call to :meth:`.InstructionSet.c_if` propagate through
        the context managers correctly."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]
        bits = qubits + clbits
        cond = (clbits[0], 0)

        with self.subTest("if"):
            test = QuantumCircuit(bits)
            with test.if_test(cond):
                test.h(0).c_if(1, 0)

            body = QuantumCircuit([qubits[0]], clbits)
            body.h(qubits[0]).c_if(clbits[1], 0)
            expected = QuantumCircuit(bits)
            expected.if_test(cond, body, [qubits[0]], clbits)

        with self.subTest("else"):
            test = QuantumCircuit(bits)
            with test.if_test(cond) as else_:
                pass
            with else_:
                test.h(0).c_if(1, 0)

            true_body = QuantumCircuit([qubits[0]], clbits)
            false_body = QuantumCircuit([qubits[0]], clbits)
            false_body.h(qubits[0]).c_if(clbits[1], 0)
            expected = QuantumCircuit(bits)
            expected.if_else(cond, true_body, false_body, [qubits[0]], clbits)

        with self.subTest("for"):
            test = QuantumCircuit(bits)
            with test.for_loop(range(2)):
                test.h(0).c_if(1, 0)

            body = QuantumCircuit([qubits[0]], clbits)
            body.h(qubits[0]).c_if(clbits[1], 0)
            expected = QuantumCircuit(bits)
            expected.for_loop(range(2), None, body, [qubits[0]], clbits)

        with self.subTest("while"):
            test = QuantumCircuit(bits)
            with test.while_loop(cond):
                test.h(0).c_if(1, 0)

            body = QuantumCircuit([qubits[0]], clbits)
            body.h(qubits[0]).c_if(clbits[1], 0)
            expected = QuantumCircuit(bits)
            expected.while_loop(cond, body, [qubits[0]], clbits)

        with self.subTest("if inside for"):
            test = QuantumCircuit(bits)
            with test.for_loop(range(2)):
                with test.if_test(cond):
                    test.h(0).c_if(1, 0)

            true_body = QuantumCircuit([qubits[0]], clbits)
            true_body.h(qubits[0]).c_if(clbits[1], 0)
            body = QuantumCircuit([qubits[0]], clbits)
            body.if_test(cond, body, [qubits[0]], clbits)
            expected = QuantumCircuit(bits)
            expected.for_loop(range(2), None, body, [qubits[0]], clbits)

    def test_access_of_classicalregister_from_c_if(self):
        """Test that resources added from a call to :meth:`.InstructionSet.c_if` propagate through
        the context managers correctly."""
        qubits = [Qubit(), Qubit()]
        creg = ClassicalRegister(2)
        clbits = [Clbit()]
        all_clbits = list(clbits) + list(creg)
        cond = (clbits[0], 0)

        with self.subTest("if"):
            test = QuantumCircuit(qubits, clbits, creg)
            with test.if_test(cond):
                test.h(0).c_if(creg, 0)

            body = QuantumCircuit([qubits[0]], clbits, creg)
            body.h(qubits[0]).c_if(creg, 0)
            expected = QuantumCircuit(qubits, clbits, creg)
            expected.if_test(cond, body, [qubits[0]], all_clbits)

        with self.subTest("else"):
            test = QuantumCircuit(qubits, clbits, creg)
            with test.if_test(cond) as else_:
                pass
            with else_:
                test.h(0).c_if(1, 0)

            true_body = QuantumCircuit([qubits[0]], clbits, creg)
            false_body = QuantumCircuit([qubits[0]], clbits, creg)
            false_body.h(qubits[0]).c_if(creg, 0)
            expected = QuantumCircuit(qubits, clbits, creg)
            expected.if_else(cond, true_body, false_body, [qubits[0]], all_clbits)

        with self.subTest("for"):
            test = QuantumCircuit(qubits, clbits, creg)
            with test.for_loop(range(2)):
                test.h(0).c_if(1, 0)

            body = QuantumCircuit([qubits[0]], clbits, creg)
            body.h(qubits[0]).c_if(creg, 0)
            expected = QuantumCircuit(qubits, clbits, creg)
            expected.for_loop(range(2), None, body, [qubits[0]], all_clbits)

        with self.subTest("while"):
            test = QuantumCircuit(qubits, clbits, creg)
            with test.while_loop(cond):
                test.h(0).c_if(creg, 0)

            body = QuantumCircuit([qubits[0]], clbits, creg)
            body.h(qubits[0]).c_if(creg, 0)
            expected = QuantumCircuit(qubits, clbits, creg)
            expected.while_loop(cond, body, [qubits[0]], all_clbits)

        with self.subTest("if inside for"):
            test = QuantumCircuit(qubits, clbits, creg)
            with test.for_loop(range(2)):
                with test.if_test(cond):
                    test.h(0).c_if(creg, 0)

            true_body = QuantumCircuit([qubits[0]], clbits, creg)
            true_body.h(qubits[0]).c_if(creg, 0)
            body = QuantumCircuit([qubits[0]], clbits, creg)
            body.if_test(cond, body, [qubits[0]], all_clbits)
            expected = QuantumCircuit(qubits, clbits, creg)
            expected.for_loop(range(2), None, body, [qubits[0]], all_clbits)

    def test_accept_broadcast_gates(self):
        """Test that the context managers accept gates that are broadcast during their addition to
        the scope."""
        qubits = [Qubit(), Qubit(), Qubit()]
        clbits = [Clbit(), Clbit(), Clbit()]
        cond = (clbits[0], 0)

        with self.subTest("if"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond):
                test.measure([0, 1], [0, 1])

            body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1]])
            body.measure(qubits[0], clbits[0])
            body.measure(qubits[1], clbits[1])

            expected = QuantumCircuit(qubits, clbits)
            expected.if_test(cond, body, [qubits[0], qubits[1]], [clbits[0], clbits[1]])

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("else"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond) as else_:
                pass
            with else_:
                test.measure([0, 1], [0, 1])

            true_body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1]])
            false_body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1]])
            false_body.measure(qubits[0], clbits[0])
            false_body.measure(qubits[1], clbits[1])

            expected = QuantumCircuit(qubits, clbits)
            expected.if_else(
                cond, true_body, false_body, [qubits[0], qubits[1]], [clbits[0], clbits[1]]
            )

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("for"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                test.measure([0, 1], [0, 1])

            body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1]])
            body.measure(qubits[0], clbits[0])
            body.measure(qubits[1], clbits[1])

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(range(2), None, body, [qubits[0], qubits[1]], [clbits[0], clbits[1]])

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("while"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond):
                test.measure([0, 1], [0, 1])

            body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1]])
            body.measure(qubits[0], clbits[0])
            body.measure(qubits[1], clbits[1])

            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(cond, body, [qubits[0], qubits[1]], [clbits[0], clbits[1]])

            self.assertCircuitsEquivalent(test, expected)

        with self.subTest("if inside for"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                with test.if_test(cond):
                    test.measure([0, 1], [0, 1])

            true_body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1]])
            true_body.measure(qubits[0], clbits[0])
            true_body.measure(qubits[1], clbits[1])

            for_body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1]])
            for_body.if_test(cond, true_body, [qubits[0], qubits[1]], [clbits[0], clbits[1]])

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(
                range(2), None, for_body, [qubits[0], qubits[1]], [clbits[0], clbits[1]]
            )

            self.assertCircuitsEquivalent(test, expected)

    def test_labels_propagated_to_instruction(self):
        """Test that labels given to the circuit-builder interface are passed through."""
        bits = [Qubit(), Clbit()]
        cond = (bits[1], 0)
        label = "sentinel_label"

        with self.subTest("if"):
            test = QuantumCircuit(bits)
            with test.if_test(cond, label=label):
                pass
            instruction = test.data[-1][0]
            self.assertIsInstance(instruction, IfElseOp)
            self.assertEqual(instruction.label, label)

        with self.subTest("if else"):
            test = QuantumCircuit(bits)
            with test.if_test(cond, label=label) as else_:
                pass
            with else_:
                pass
            instruction = test.data[-1][0]
            self.assertIsInstance(instruction, IfElseOp)
            self.assertEqual(instruction.label, label)

        with self.subTest("for"):
            test = QuantumCircuit(bits)
            with test.for_loop(range(2), label=label):
                pass
            instruction = test.data[-1][0]
            self.assertIsInstance(instruction, ForLoopOp)
            self.assertEqual(instruction.label, label)

        with self.subTest("while"):
            test = QuantumCircuit(bits)
            with test.while_loop(cond, label=label):
                pass
            instruction = test.data[-1][0]
            self.assertIsInstance(instruction, WhileLoopOp)
            self.assertEqual(instruction.label, label)

        # The tests of 'if' and 'else' inside 'for' are to ensure we're hitting the paths where the
        # 'if' scope is built lazily at the completion of the 'for'.
        with self.subTest("if inside for"):
            test = QuantumCircuit(bits)
            with test.for_loop(range(2)):
                with test.if_test(cond, label=label):
                    # Use break to ensure that we're triggering the lazy building of 'if'.
                    test.break_loop()

            instruction = test.data[-1][0].blocks[0].data[-1][0]
            self.assertIsInstance(instruction, IfElseOp)
            self.assertEqual(instruction.label, label)

        with self.subTest("else inside for"):
            test = QuantumCircuit(bits)
            with test.for_loop(range(2)):
                with test.if_test(cond, label=label) as else_:
                    # Use break to ensure that we're triggering the lazy building of 'if'.
                    test.break_loop()
                with else_:
                    test.break_loop()

            instruction = test.data[-1][0].blocks[0].data[-1][0]
            self.assertIsInstance(instruction, IfElseOp)
            self.assertEqual(instruction.label, label)

    def test_copy_of_circuits(self):
        """Test that various methods of copying a circuit made with the builder interface works."""
        test = QuantumCircuit(5, 5)
        cond = (test.clbits[2], False)
        with test.if_test(cond) as else_:
            test.cx(0, 1)
        with else_:
            test.h(2)
        with test.for_loop(range(5)):
            with test.if_test(cond):
                test.x(3)
        with test.while_loop(cond):
            test.measure(0, 4)
        self.assertEqual(test, test.copy())
        self.assertEqual(test, copy.copy(test))
        self.assertEqual(test, copy.deepcopy(test))

    def test_copy_of_instructions(self):
        """Test that various methods of copying the actual instructions created by the builder
        interface work."""
        qubits = [Qubit() for _ in [None] * 3]
        clbits = [Clbit() for _ in [None] * 3]
        cond = (clbits[1], False)

        with self.subTest("if"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond):
                test.cx(0, 1)
                test.measure(2, 2)
            if_instruction, _, _ = test.data[0]
            self.assertEqual(if_instruction, if_instruction.copy())
            self.assertEqual(if_instruction, copy.copy(if_instruction))
            self.assertEqual(if_instruction, copy.deepcopy(if_instruction))

        with self.subTest("if/else"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond) as else_:
                test.cx(0, 1)
                test.measure(2, 2)
            with else_:
                test.cx(1, 0)
                test.measure(2, 2)
            if_instruction, _, _ = test.data[0]
            self.assertEqual(if_instruction, if_instruction.copy())
            self.assertEqual(if_instruction, copy.copy(if_instruction))
            self.assertEqual(if_instruction, copy.deepcopy(if_instruction))

        with self.subTest("for"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(4)):
                test.cx(0, 1)
                test.measure(2, 2)
            for_instruction, _, _ = test.data[0]
            self.assertEqual(for_instruction, for_instruction.copy())
            self.assertEqual(for_instruction, copy.copy(for_instruction))
            self.assertEqual(for_instruction, copy.deepcopy(for_instruction))

        with self.subTest("while"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond):
                test.cx(0, 1)
                test.measure(2, 2)
            while_instruction, _, _ = test.data[0]
            self.assertEqual(while_instruction, while_instruction.copy())
            self.assertEqual(while_instruction, copy.copy(while_instruction))
            self.assertEqual(while_instruction, copy.deepcopy(while_instruction))

    def test_copy_of_instruction_parameters(self):
        """Test that various methods of copying the parameters inside instructions created by the
        builder interface work.  Regression test of gh-7367."""
        qubits = [Qubit() for _ in [None] * 3]
        clbits = [Clbit() for _ in [None] * 3]
        cond = (clbits[1], False)

        with self.subTest("if"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond):
                test.cx(0, 1)
                test.measure(2, 2)
            if_instruction, _, _ = test.data[0]
            (true_body,) = if_instruction.blocks
            self.assertEqual(true_body, true_body.copy())
            self.assertEqual(true_body, copy.copy(true_body))
            self.assertEqual(true_body, copy.deepcopy(true_body))

        with self.subTest("if/else"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond) as else_:
                test.cx(0, 1)
                test.measure(2, 2)
            with else_:
                test.cx(1, 0)
                test.measure(2, 2)
            if_instruction, _, _ = test.data[0]
            true_body, false_body = if_instruction.blocks
            self.assertEqual(true_body, true_body.copy())
            self.assertEqual(true_body, copy.copy(true_body))
            self.assertEqual(true_body, copy.deepcopy(true_body))
            self.assertEqual(false_body, false_body.copy())
            self.assertEqual(false_body, copy.copy(false_body))
            self.assertEqual(false_body, copy.deepcopy(false_body))

        with self.subTest("for"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(4)):
                test.cx(0, 1)
                test.measure(2, 2)
            for_instruction, _, _ = test.data[0]
            (for_body,) = for_instruction.blocks
            self.assertEqual(for_body, for_body.copy())
            self.assertEqual(for_body, copy.copy(for_body))
            self.assertEqual(for_body, copy.deepcopy(for_body))

        with self.subTest("while"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond):
                test.cx(0, 1)
                test.measure(2, 2)
            while_instruction, _, _ = test.data[0]
            (while_body,) = while_instruction.blocks
            self.assertEqual(while_body, while_body.copy())
            self.assertEqual(while_body, copy.copy(while_body))
            self.assertEqual(while_body, copy.deepcopy(while_body))


@ddt.ddt
class TestControlFlowBuildersFailurePaths(QiskitTestCase):
    """Tests for the failure paths of the control-flow builders."""

    def test_if_rejects_break_continue_if_not_in_loop(self):
        """Test that the ``if`` and ``else`` context managers raise a suitable exception if you try
        to use a ``break`` or ``continue`` within them without being inside a loop.  This is for
        safety; without the loop context, the context manager will cause the wrong resources to be
        assigned to the ``break``, so if you want to make a manual loop, you have to use manual
        ``if`` as well.  That way the onus is on you."""
        qubits = [Qubit()]
        clbits = [Clbit()]
        cond = (clbits[0], 0)

        message = r"The current builder scope cannot take a '.*' because it is not in a loop\."

        with self.subTest("if break"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond):
                with self.assertRaisesRegex(CircuitError, message):
                    test.break_loop()

        with self.subTest("if continue"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond):
                with self.assertRaisesRegex(CircuitError, message):
                    test.continue_loop()

        with self.subTest("else break"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond) as else_:
                pass
            with else_:
                with self.assertRaisesRegex(CircuitError, message):
                    test.break_loop()

        with self.subTest("else continue"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond) as else_:
                pass
            with else_:
                with self.assertRaisesRegex(CircuitError, message):
                    test.continue_loop()

    def test_for_rejects_reentry(self):
        """Test that the ``for``-loop context manager rejects attempts to re-enter it.  Since it
        holds some forms of state during execution (the loop variable, which may be generated), we
        can't safely re-enter it and get the expected behaviour."""

        for_manager = QuantumCircuit(2, 2).for_loop(range(2))
        with for_manager:
            pass
        with self.assertRaisesRegex(
            CircuitError, r"A for-loop context manager cannot be re-entered."
        ):
            with for_manager:
                pass

    def test_cannot_enter_else_context_incorrectly(self):
        """Test that various forms of using an 'else_' context manager incorrectly raise
        exceptions."""
        bits = [Qubit(), Clbit()]
        cond = (bits[1], 0)

        with self.subTest("not the next instruction"):
            test = QuantumCircuit(bits)
            with test.if_test(cond) as else_:
                pass
            test.h(0)
            with self.assertRaisesRegex(CircuitError, "The 'if' block is not the most recent"):
                with else_:
                    test.h(0)

        with self.subTest("inside the attached if"):
            test = QuantumCircuit(bits)
            with test.if_test(cond) as else_:
                with self.assertRaisesRegex(
                    CircuitError, r"Cannot attach an 'else' branch to an incomplete 'if' block\."
                ):
                    with else_:
                        test.h(0)

        with self.subTest("inner else"):
            test = QuantumCircuit(bits)
            with test.if_test(cond) as else1:
                with test.if_test(cond):
                    pass
                with self.assertRaisesRegex(
                    CircuitError, r"Cannot attach an 'else' branch to an incomplete 'if' block\."
                ):
                    with else1:
                        test.h(0)

        with self.subTest("reused else"):
            test = QuantumCircuit(bits)
            with test.if_test(cond) as else_:
                pass
            with else_:
                pass
            with self.assertRaisesRegex(CircuitError, r"Cannot re-use an 'else' context\."):
                with else_:
                    pass

        with self.subTest("else from an inner block"):
            test = QuantumCircuit(bits)
            with test.if_test(cond):
                with test.if_test(cond) as else_:
                    pass
            with self.assertRaisesRegex(CircuitError, "The 'if' block is not the most recent"):
                with else_:
                    pass

    def test_if_placeholder_rejects_c_if(self):
        """Test that the :obj:`.IfElsePlaceholder" class rejects attempts to use
        :meth:`.Instruction.c_if` on it.

        It *should* be the case that you need to use private methods to get access to one of these
        placeholder objects at all, because they're appended to a scope at the exit of a context
        manager, so not returned from a method call. Just in case, here's a test that it correctly
        rejects the dangerous method that can overwrite ``condition``.
        """
        bits = [Qubit(), Clbit()]

        with self.subTest("if"):
            test = QuantumCircuit(bits)
            with test.for_loop(range(2)):
                with test.if_test((bits[1], 0)):
                    test.break_loop()
                # These tests need to be done before the 'for' context exits so we don't trigger the
                # "can't add conditions from out-of-scope" handling.
                placeholder, _, _ = test._peek_previous_instruction_in_scope()
                self.assertIsInstance(placeholder, IfElsePlaceholder)
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"IfElseOp cannot be classically controlled through Instruction\.c_if",
                ):
                    placeholder.c_if(bits[1], 0)

        with self.subTest("else"):
            test = QuantumCircuit(bits)
            with test.for_loop(range(2)):
                with test.if_test((bits[1], 0)) as else_:
                    pass
                with else_:
                    test.break_loop()
                # These tests need to be done before the 'for' context exits so we don't trigger the
                # "can't add conditions from out-of-scope" handling.
                placeholder, _, _ = test._peek_previous_instruction_in_scope()
                self.assertIsInstance(placeholder, IfElsePlaceholder)
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"IfElseOp cannot be classically controlled through Instruction\.c_if",
                ):
                    placeholder.c_if(bits[1], 0)

    def test_reject_c_if_from_outside_scope(self):
        """Test that the context managers reject :meth:`.InstructionSet.c_if` calls if they occur
        after their scope has completed."""
        bits = [Qubit(), Clbit()]
        cond = (bits[1], 0)

        with self.subTest("if"):
            test = QuantumCircuit(bits)
            with test.if_test(cond):
                instructions = test.h(0)
            with self.assertRaisesRegex(
                CircuitError, r"Cannot add resources after the scope has been built\."
            ):
                instructions.c_if(*cond)

        with self.subTest("else"):
            test = QuantumCircuit(bits)
            with test.if_test(cond) as else_:
                pass
            with else_:
                instructions = test.h(0)
            with self.assertRaisesRegex(
                CircuitError, r"Cannot add resources after the scope has been built\."
            ):
                instructions.c_if(*cond)

        with self.subTest("for"):
            test = QuantumCircuit(bits)
            with test.for_loop(range(2)):
                instructions = test.h(0)
            with self.assertRaisesRegex(
                CircuitError, r"Cannot add resources after the scope has been built\."
            ):
                instructions.c_if(*cond)

        with self.subTest("while"):
            test = QuantumCircuit(bits)
            with test.while_loop(cond):
                instructions = test.h(0)
            with self.assertRaisesRegex(
                CircuitError, r"Cannot add resources after the scope has been built\."
            ):
                instructions.c_if(*cond)

        with self.subTest("if inside for"):
            # As a side-effect of how the lazy building of 'if' statements works, we actually
            # *could* add a condition to the gate after the 'if' block as long as we were still
            # within the 'for' loop.  It should actually manage the resource correctly as well, but
            # it's "undefined behaviour" than something we specifically want to forbid or allow.
            test = QuantumCircuit(bits)
            with test.for_loop(range(2)):
                with test.if_test(cond):
                    instructions = test.h(0)
            with self.assertRaisesRegex(
                CircuitError, r"Cannot add resources after the scope has been built\."
            ):

                instructions.c_if(*cond)

    def test_raising_inside_context_manager_leave_circuit_usable(self):
        """Test that if we leave a builder by raising some sort of exception, the circuit is left in
        a usable state, and extra resources have not been added to the circuit."""

        x, y = Parameter("x"), Parameter("y")

        with self.subTest("for"):
            test = QuantumCircuit(1, 1)
            test.h(0)
            with self.assertRaises(SentinelException):
                with test.for_loop(range(2), x) as bound_x:
                    test.x(0)
                    test.rx(bound_x, 0)
                    test.ry(y, 0)
                    raise SentinelException
            test.z(0)

            expected = QuantumCircuit(1, 1)
            expected.h(0)
            expected.z(0)

            self.assertEqual(test, expected)
            # We don't want _either_ the loop variable or the loose variable to be in the circuit.
            self.assertEqual(set(), set(test.parameters))

        with self.subTest("while"):
            bits = [Qubit(), Clbit()]
            test = QuantumCircuit(bits)
            test.h(0)
            with self.assertRaises(SentinelException):
                with test.while_loop((bits[1], 0)):
                    test.x(0)
                    test.rx(x, 0)
                    raise SentinelException
            test.z(0)

            expected = QuantumCircuit(bits)
            expected.h(0)
            expected.z(0)

            self.assertEqual(test, expected)
            self.assertEqual(set(), set(test.parameters))

        with self.subTest("if"):
            bits = [Qubit(), Clbit()]
            test = QuantumCircuit(bits)
            test.h(0)
            with self.assertRaises(SentinelException):
                with test.if_test((bits[1], 0)):
                    test.x(0)
                    test.rx(x, 0)
                    raise SentinelException
            test.z(0)

            expected = QuantumCircuit(bits)
            expected.h(0)
            expected.z(0)

            self.assertEqual(test, expected)
            self.assertEqual(set(), set(test.parameters))

        with self.subTest("else"):
            bits = [Qubit(), Clbit()]
            test = QuantumCircuit(bits)
            test.h(0)
            with test.if_test((bits[1], 0)) as else_:
                test.rx(x, 0)
            with self.assertRaises(SentinelException):
                with else_:
                    test.x(0)
                    test.rx(y, 0)
                    raise SentinelException
            test.z(0)

            # Note that we expect the "else" manager to restore the "if" block if something errors
            # out during "else" block.
            true_body = QuantumCircuit(bits)
            true_body.rx(x, 0)
            expected = QuantumCircuit(bits)
            expected.h(0)
            expected.if_test((bits[1], 0), true_body, [0], [0])
            expected.z(0)

            self.assertEqual(test, expected)
            self.assertEqual({x}, set(test.parameters))

    def test_can_reuse_else_manager_after_exception(self):
        """Test that the "else" context manager is usable after a first attempt to construct it
        raises an exception.  Normally you cannot re-enter an "else" block, but we want the user to
        be able to recover from errors if they so try."""
        bits = [Qubit(), Clbit()]
        test = QuantumCircuit(bits)
        test.h(0)
        with test.if_test((bits[1], 0)) as else_:
            test.x(0)
        with self.assertRaises(SentinelException):
            with else_:
                test.y(0)
                raise SentinelException
        with else_:
            test.h(0)
        test.z(0)

        true_body = QuantumCircuit(bits)
        true_body.x(0)
        false_body = QuantumCircuit(bits)
        false_body.h(0)
        expected = QuantumCircuit(bits)
        expected.h(0)
        expected.if_else((bits[1], 0), true_body, false_body, [0], [0])
        expected.z(0)

        self.assertEqual(test, expected)

    @ddt.data((None, [0]), ([0], None), ([0], [0]))
    def test_context_managers_reject_passing_qubits(self, resources):
        """Test that the context-manager forms of the control-flow circuit methods raise exceptions
        if they are given explicit qubits or clbits."""
        test = QuantumCircuit(1, 1)
        qubits, clbits = resources
        with self.subTest("for"):
            with self.assertRaisesRegex(
                CircuitError,
                r"When using 'for_loop' as a context manager, you cannot pass qubits or clbits\.",
            ):
                test.for_loop(range(2), None, body=None, qubits=qubits, clbits=clbits)
        with self.subTest("while"):
            with self.assertRaisesRegex(
                CircuitError,
                r"When using 'while_loop' as a context manager, you cannot pass qubits or clbits\.",
            ):
                test.while_loop((test.clbits[0], 0), body=None, qubits=qubits, clbits=clbits)
        with self.subTest("if"):
            with self.assertRaisesRegex(
                CircuitError,
                r"When using 'if_test' as a context manager, you cannot pass qubits or clbits\.",
            ):
                test.if_test((test.clbits[0], 0), true_body=None, qubits=qubits, clbits=clbits)

    @ddt.data((None, [0]), ([0], None), (None, None))
    def test_non_context_manager_calling_states_reject_missing_resources(self, resources):
        """Test that the non-context-manager forms of the control-flow circuit methods raise
        exceptions if they are not given explicit qubits or clbits."""
        test = QuantumCircuit(1, 1)
        body = QuantumCircuit(1, 1)
        qubits, clbits = resources
        with self.subTest("for"):
            with self.assertRaisesRegex(
                CircuitError,
                r"When using 'for_loop' with a body, you must pass qubits and clbits\.",
            ):
                test.for_loop(range(2), None, body=body, qubits=qubits, clbits=clbits)
        with self.subTest("while"):
            with self.assertRaisesRegex(
                CircuitError,
                r"When using 'while_loop' with a body, you must pass qubits and clbits\.",
            ):
                test.while_loop((test.clbits[0], 0), body=body, qubits=qubits, clbits=clbits)
        with self.subTest("if"):
            with self.assertRaisesRegex(
                CircuitError,
                r"When using 'if_test' with a body, you must pass qubits and clbits\.",
            ):
                test.if_test((test.clbits[0], 0), true_body=body, qubits=qubits, clbits=clbits)

    @ddt.data(None, [Clbit()], 0)
    def test_builder_block_add_bits_reject_bad_bits(self, bit):
        """Test that :obj:`.ControlFlowBuilderBlock` raises if something is given that is an
        incorrect type.

        This isn't intended to be something users do at all; the builder block is an internal
        construct only, but this keeps coverage checking happy."""

        def dummy_requester(resource):
            raise CircuitError

        builder_block = ControlFlowBuilderBlock(
            qubits=(), clbits=(), resource_requester=dummy_requester
        )
        with self.assertRaisesRegex(TypeError, r"Can only add qubits or classical bits.*"):
            builder_block.add_bits([bit])
