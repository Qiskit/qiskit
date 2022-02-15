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

"""Test operations on control flow for dynamic QuantumCircuits."""

import math

from ddt import ddt, data

from qiskit.test import QiskitTestCase
from qiskit.circuit import Clbit, ClassicalRegister, Instruction, Parameter, QuantumCircuit
from qiskit.circuit.library import XGate, RXGate
from qiskit.circuit.exceptions import CircuitError

from qiskit.circuit.controlflow import (
    ControlFlowOp,
    WhileLoopOp,
    ForLoopOp,
    IfElseOp,
    ContinueLoopOp,
    BreakLoopOp,
)


@ddt
class TestCreatingControlFlowOperations(QiskitTestCase):
    """Tests instantiation of instruction subclasses for dynamic QuantumCircuits."""

    @data(
        (Clbit(), True),
        (ClassicalRegister(3, "test_creg"), 3),
        (ClassicalRegister(3, "test_creg"), True),
    )
    def test_while_loop_instantiation(self, condition):
        """Verify creation and properties of a WhileLoopOp."""
        body = QuantumCircuit(3, 1)
        body.add_register([condition[0]] if isinstance(condition[0], Clbit) else condition[0])

        op = WhileLoopOp(condition, body)

        self.assertIsInstance(op, ControlFlowOp)
        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "while_loop")
        self.assertEqual(op.num_qubits, body.num_qubits)
        self.assertEqual(op.num_clbits, body.num_clbits)
        self.assertEqual(op.condition, condition)
        self.assertEqual(op.params, [body])
        self.assertEqual(op.blocks, (body,))

    def test_while_loop_invalid_instantiation(self):
        """Verify we catch invalid instantiations of WhileLoopOp."""
        body = QuantumCircuit(3, 1)
        condition = (body.clbits[0], True)

        with self.assertRaisesRegex(CircuitError, r"A classical condition should be a 2-tuple"):
            _ = WhileLoopOp(0, body)

        with self.assertRaisesRegex(CircuitError, r"A classical condition should be a 2-tuple"):
            _ = WhileLoopOp((Clbit(), None), body)

        with self.assertRaisesRegex(CircuitError, r"of type QuantumCircuit"):
            _ = WhileLoopOp(condition, XGate())

    def test_while_loop_invalid_params_setter(self):
        """Verify we catch invalid param settings for WhileLoopOp."""
        body = QuantumCircuit(3, 1)
        condition = (body.clbits[0], True)

        bad_body = QuantumCircuit(2, 1)
        op = WhileLoopOp(condition, body)
        with self.assertRaisesRegex(
            CircuitError, r"num_clbits different than that of the WhileLoopOp"
        ):
            op.params = [bad_body]

    def test_for_loop_iterable_instantiation(self):
        """Verify creation and properties of a ForLoopOp using an iterable indexset."""
        body = QuantumCircuit(3, 1)
        loop_parameter = Parameter("foo")
        indexset = iter(range(0, 10, 2))

        body.rx(loop_parameter, 0)

        op = ForLoopOp(indexset, loop_parameter, body)

        self.assertIsInstance(op, ControlFlowOp)
        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "for_loop")
        self.assertEqual(op.num_qubits, 3)
        self.assertEqual(op.num_clbits, 1)
        self.assertEqual(op.params, [tuple(range(0, 10, 2)), loop_parameter, body])
        self.assertEqual(op.blocks, (body,))

    def test_for_loop_range_instantiation(self):
        """Verify creation and properties of a ForLoopOp using a range indexset."""
        body = QuantumCircuit(3, 1)
        loop_parameter = Parameter("foo")
        indexset = range(0, 10, 2)

        body.rx(loop_parameter, 0)

        op = ForLoopOp(indexset, loop_parameter, body)

        self.assertIsInstance(op, ControlFlowOp)
        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "for_loop")
        self.assertEqual(op.num_qubits, 3)
        self.assertEqual(op.num_clbits, 1)
        self.assertEqual(op.params, [indexset, loop_parameter, body])
        self.assertEqual(op.blocks, (body,))

    def test_for_loop_no_parameter_instantiation(self):
        """Verify creation and properties of a ForLoopOp without a loop_parameter."""
        body = QuantumCircuit(3, 1)
        loop_parameter = None
        indexset = range(0, 10, 2)

        body.rx(3.14, 0)

        op = ForLoopOp(indexset, loop_parameter, body)

        self.assertIsInstance(op, ControlFlowOp)
        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "for_loop")
        self.assertEqual(op.num_qubits, 3)
        self.assertEqual(op.num_clbits, 1)
        self.assertEqual(op.params, [indexset, loop_parameter, body])
        self.assertEqual(op.blocks, (body,))

    def test_for_loop_invalid_instantiation(self):
        """Verify we catch invalid instantiations of ForLoopOp."""
        body = QuantumCircuit(3, 1)
        loop_parameter = Parameter("foo")
        indexset = range(0, 10, 2)

        body.rx(loop_parameter, 0)

        with self.assertWarnsRegex(UserWarning, r"loop_parameter was not found"):
            _ = ForLoopOp(indexset, Parameter("foo"), body)

        with self.assertRaisesRegex(CircuitError, r"to be of type QuantumCircuit"):
            _ = ForLoopOp(indexset, loop_parameter, RXGate(loop_parameter))

    def test_for_loop_invalid_params_setter(self):
        """Verify we catch invalid param settings for ForLoopOp."""
        body = QuantumCircuit(3, 1)
        loop_parameter = Parameter("foo")
        indexset = range(0, 10, 2)

        body.rx(loop_parameter, 0)

        op = ForLoopOp(indexset, loop_parameter, body)

        with self.assertWarnsRegex(UserWarning, r"loop_parameter was not found"):
            op.params = [indexset, Parameter("foo"), body]

        with self.assertRaisesRegex(CircuitError, r"to be of type QuantumCircuit"):
            op.params = [indexset, loop_parameter, RXGate(loop_parameter)]

        bad_body = QuantumCircuit(2, 1)
        with self.assertRaisesRegex(
            CircuitError, r"num_clbits different than that of the ForLoopOp"
        ):
            op.params = [indexset, loop_parameter, bad_body]

        with self.assertRaisesRegex(CircuitError, r"to be either of type Parameter or None"):
            _ = ForLoopOp(indexset, "foo", body)

    @data(
        (Clbit(), True),
        (ClassicalRegister(3, "test_creg"), 3),
        (ClassicalRegister(3, "test_creg"), True),
    )
    def test_if_else_instantiation_with_else(self, condition):
        """Verify creation and properties of a IfElseOp with an else branch."""
        true_body = QuantumCircuit(3, 1)
        false_body = QuantumCircuit(3, 1)

        op = IfElseOp(condition, true_body, false_body)

        self.assertIsInstance(op, ControlFlowOp)
        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "if_else")
        self.assertEqual(op.num_qubits, 3)
        self.assertEqual(op.num_clbits, 1)
        self.assertEqual(op.params, [true_body, false_body])
        self.assertEqual(op.condition, condition)
        self.assertEqual(op.blocks, (true_body, false_body))

    @data(
        (Clbit(), True),
        (ClassicalRegister(3, "test_creg"), 3),
        (ClassicalRegister(3, "test_creg"), True),
    )
    def test_if_else_instantiation_without_else(self, condition):
        """Verify creation and properties of a IfElseOp without an else branch."""
        true_body = QuantumCircuit(3, 1)

        op = IfElseOp(condition, true_body)

        self.assertIsInstance(op, ControlFlowOp)
        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "if_else")
        self.assertEqual(op.num_qubits, 3)
        self.assertEqual(op.num_clbits, 1)
        self.assertEqual(op.params, [true_body, None])
        self.assertEqual(op.condition, condition)
        self.assertEqual(op.blocks, (true_body,))

    def test_if_else_invalid_instantiation(self):
        """Verify we catch invalid instantiations of IfElseOp."""
        condition = (Clbit(), True)
        true_body = QuantumCircuit(3, 1)
        false_body = QuantumCircuit(3, 1)

        with self.assertRaisesRegex(CircuitError, r"A classical condition should be a 2-tuple"):
            _ = IfElseOp(1, true_body, false_body)

        with self.assertRaisesRegex(CircuitError, r"A classical condition should be a 2-tuple"):
            _ = IfElseOp((1, 2), true_body, false_body)

        with self.assertRaisesRegex(CircuitError, r"true_body parameter of type QuantumCircuit"):
            _ = IfElseOp(condition, XGate())

        with self.assertRaisesRegex(CircuitError, r"false_body parameter of type QuantumCircuit"):
            _ = IfElseOp(condition, true_body, XGate())

        bad_body = QuantumCircuit(4, 2)

        with self.assertRaisesRegex(
            CircuitError, r"num_clbits different than that of the IfElseOp"
        ):
            _ = IfElseOp(condition, true_body, bad_body)

    def test_if_else_invalid_params_setter(self):
        """Verify we catch invalid param settings for IfElseOp."""
        condition = (Clbit(), True)
        true_body = QuantumCircuit(3, 1)
        false_body = QuantumCircuit(3, 1)

        op = IfElseOp(condition, true_body, false_body)

        with self.assertRaisesRegex(CircuitError, r"true_body parameter of type QuantumCircuit"):
            op.params = [XGate(), None]

        with self.assertRaisesRegex(CircuitError, r"false_body parameter of type QuantumCircuit"):
            op.params = [true_body, XGate()]

        bad_body = QuantumCircuit(4, 2)

        with self.assertRaisesRegex(
            CircuitError, r"num_clbits different than that of the IfElseOp"
        ):
            op.params = [true_body, bad_body]

        with self.assertRaisesRegex(
            CircuitError, r"num_clbits different than that of the IfElseOp"
        ):
            op.params = [bad_body, false_body]

        with self.assertRaisesRegex(
            CircuitError, r"num_clbits different than that of the IfElseOp"
        ):
            op.params = [bad_body, bad_body]

    def test_continue_loop_instantiation(self):
        """Verify creation and properties of a ContinueLoopOp."""
        op = ContinueLoopOp(3, 1)

        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "continue_loop")
        self.assertEqual(op.num_qubits, 3)
        self.assertEqual(op.num_clbits, 1)
        self.assertEqual(op.params, [])

    def test_break_loop_instantiation(self):
        """Verify creation and properties of a BreakLoopOp."""
        op = BreakLoopOp(3, 1)

        self.assertIsInstance(op, Instruction)
        self.assertEqual(op.name, "break_loop")
        self.assertEqual(op.num_qubits, 3)
        self.assertEqual(op.num_clbits, 1)
        self.assertEqual(op.params, [])


@ddt
class TestAddingControlFlowOperations(QiskitTestCase):
    """Tests of instruction subclasses for dynamic QuantumCircuits."""

    @data(
        (Clbit(), True),
        (ClassicalRegister(3, "test_creg"), 3),
        (ClassicalRegister(3, "test_creg"), True),
    )
    def test_appending_while_loop_op(self, condition):
        """Verify we can append a WhileLoopOp to a QuantumCircuit."""
        body = QuantumCircuit(3, 1)
        body.add_register([condition[0]] if isinstance(condition[0], Clbit) else condition[0])

        op = WhileLoopOp(condition, body)

        qc = QuantumCircuit(5, 2)
        qc.append(op, [1, 2, 3], [1])

        self.assertEqual(qc.data[0][0].name, "while_loop")
        self.assertEqual(qc.data[0][0].params, [body])
        self.assertEqual(qc.data[0][0].condition, condition)
        self.assertEqual(qc.data[0][1], qc.qubits[1:4])
        self.assertEqual(qc.data[0][2], [qc.clbits[1]])

    @data(
        (Clbit(), True),
        (ClassicalRegister(3, "test_creg"), 3),
        (ClassicalRegister(3, "test_creg"), True),
    )
    def test_quantumcircuit_while_loop(self, condition):
        """Verify we can append a WhileLoopOp to a QuantumCircuit via qc.while_loop."""
        body = QuantumCircuit(3, 1)
        body.add_register([condition[0]] if isinstance(condition[0], Clbit) else condition[0])

        qc = QuantumCircuit(5, 2)
        qc.while_loop(condition, body, [1, 2, 3], [1])

        self.assertEqual(qc.data[0][0].name, "while_loop")
        self.assertEqual(qc.data[0][0].params, [body])
        self.assertEqual(qc.data[0][0].condition, condition)
        self.assertEqual(qc.data[0][1], qc.qubits[1:4])
        self.assertEqual(qc.data[0][2], [qc.clbits[1]])

    def test_appending_for_loop_op(self):
        """Verify we can append a ForLoopOp to a QuantumCircuit."""
        body = QuantumCircuit(3, 1)
        loop_parameter = Parameter("foo")
        indexset = range(0, 10, 2)

        body.rx(loop_parameter, [0, 1, 2])

        op = ForLoopOp(indexset, loop_parameter, body)

        qc = QuantumCircuit(5, 2)
        qc.append(op, [1, 2, 3], [1])

        self.assertEqual(qc.data[0][0].name, "for_loop")
        self.assertEqual(qc.data[0][0].params, [indexset, loop_parameter, body])
        self.assertEqual(qc.data[0][1], qc.qubits[1:4])
        self.assertEqual(qc.data[0][2], [qc.clbits[1]])

    def test_quantumcircuit_for_loop_op(self):
        """Verify we can append a ForLoopOp to a QuantumCircuit via qc.for_loop."""
        body = QuantumCircuit(3, 1)
        loop_parameter = Parameter("foo")
        indexset = range(0, 10, 2)

        body.rx(loop_parameter, [0, 1, 2])

        qc = QuantumCircuit(5, 2)
        qc.for_loop(indexset, loop_parameter, body, [1, 2, 3], [1])

        self.assertEqual(qc.data[0][0].name, "for_loop")
        self.assertEqual(qc.data[0][0].params, [indexset, loop_parameter, body])
        self.assertEqual(qc.data[0][1], qc.qubits[1:4])
        self.assertEqual(qc.data[0][2], [qc.clbits[1]])

    @data(
        (Clbit(), True),
        (ClassicalRegister(3, "test_creg"), 3),
        (ClassicalRegister(3, "test_creg"), True),
    )
    def test_appending_if_else_op(self, condition):
        """Verify we can append a IfElseOp to a QuantumCircuit."""
        true_body = QuantumCircuit(3, 1)
        false_body = QuantumCircuit(3, 1)

        op = IfElseOp(condition, true_body, false_body)

        qc = QuantumCircuit(5, 2)
        qc.add_register([condition[0]] if isinstance(condition[0], Clbit) else condition[0])
        qc.append(op, [1, 2, 3], [1])

        self.assertEqual(qc.data[0][0].name, "if_else")
        self.assertEqual(qc.data[0][0].params, [true_body, false_body])
        self.assertEqual(qc.data[0][0].condition, condition)
        self.assertEqual(qc.data[0][1], qc.qubits[1:4])
        self.assertEqual(qc.data[0][2], [qc.clbits[1]])

    @data(
        (Clbit(), True),
        (ClassicalRegister(3, "test_creg"), 3),
        (ClassicalRegister(3, "test_creg"), True),
    )
    def test_quantumcircuit_if_else_op(self, condition):
        """Verify we can append a IfElseOp to a QuantumCircuit via qc.if_else."""
        true_body = QuantumCircuit(3, 1)
        false_body = QuantumCircuit(3, 1)

        qc = QuantumCircuit(5, 2)
        qc.add_register([condition[0]] if isinstance(condition[0], Clbit) else condition[0])
        qc.if_else(condition, true_body, false_body, [1, 2, 3], [1])

        self.assertEqual(qc.data[0][0].name, "if_else")
        self.assertEqual(qc.data[0][0].params, [true_body, false_body])
        self.assertEqual(qc.data[0][0].condition, condition)
        self.assertEqual(qc.data[0][1], qc.qubits[1:4])
        self.assertEqual(qc.data[0][2], [qc.clbits[1]])

    @data(
        (Clbit(), True),
        (ClassicalRegister(3, "test_creg"), 3),
        (ClassicalRegister(3, "test_creg"), True),
    )
    def test_quantumcircuit_if_test_op(self, condition):
        """Verify we can append a IfElseOp to a QuantumCircuit via qc.if_test."""
        true_body = QuantumCircuit(3, 1)

        qc = QuantumCircuit(5, 2)
        qc.add_register([condition[0]] if isinstance(condition[0], Clbit) else condition[0])
        qc.if_test(condition, true_body, [1, 2, 3], [1])

        self.assertEqual(qc.data[0][0].name, "if_else")
        self.assertEqual(qc.data[0][0].params, [true_body, None])
        self.assertEqual(qc.data[0][0].condition, condition)
        self.assertEqual(qc.data[0][1], qc.qubits[1:4])
        self.assertEqual(qc.data[0][2], [qc.clbits[1]])

    @data(
        (Clbit(), True),
        (ClassicalRegister(3, "test_creg"), 3),
        (ClassicalRegister(3, "test_creg"), True),
    )
    def test_appending_if_else_op_with_condition_outside(self, condition):
        """Verify we catch if IfElseOp has a condition outside outer circuit."""
        true_body = QuantumCircuit(3, 1)
        false_body = QuantumCircuit(3, 1)

        qc = QuantumCircuit(5, 2)

        with self.assertRaisesRegex(CircuitError, r".* is not present in this circuit\."):
            qc.if_test(condition, true_body, [1, 2, 3], [1])

        with self.assertRaisesRegex(CircuitError, r".* is not present in this circuit\."):
            qc.if_else(condition, true_body, false_body, [1, 2, 3], [1])

    def test_appending_continue_loop_op(self):
        """Verify we can append a ContinueLoopOp to a QuantumCircuit."""
        op = ContinueLoopOp(3, 1)

        qc = QuantumCircuit(3, 1)
        qc.append(op, [0, 1, 2], [0])

        self.assertEqual(qc.data[0][0].name, "continue_loop")
        self.assertEqual(qc.data[0][1], qc.qubits)
        self.assertEqual(qc.data[0][2], qc.clbits)

    def test_quantumcircuit_continue_loop_op(self):
        """Verify we can append a ContinueLoopOp to a QuantumCircuit via qc.continue_loop."""
        qc = QuantumCircuit(3, 1)
        qc.continue_loop()

        self.assertEqual(qc.data[0][0].name, "continue_loop")
        self.assertEqual(qc.data[0][1], qc.qubits)
        self.assertEqual(qc.data[0][2], qc.clbits)

    def test_appending_break_loop_op(self):
        """Verify we can append a BreakLoopOp to a QuantumCircuit."""
        op = BreakLoopOp(3, 1)

        qc = QuantumCircuit(3, 1)
        qc.append(op, [0, 1, 2], [0])

        self.assertEqual(qc.data[0][0].name, "break_loop")
        self.assertEqual(qc.data[0][1], qc.qubits)
        self.assertEqual(qc.data[0][2], qc.clbits)

    def test_quantumcircuit_break_loop_op(self):
        """Verify we can append a BreakLoopOp to a QuantumCircuit via qc.break_loop."""
        qc = QuantumCircuit(3, 1)
        qc.break_loop()

        self.assertEqual(qc.data[0][0].name, "break_loop")
        self.assertEqual(qc.data[0][1], qc.qubits)
        self.assertEqual(qc.data[0][2], qc.clbits)

    def test_no_c_if_for_while_loop_if_else(self):
        """Verify we raise if a user attempts to use c_if on an op which sets .condition."""
        qc = QuantumCircuit(3, 1)
        body = QuantumCircuit(1)

        with self.assertRaisesRegex(NotImplementedError, r"cannot be classically controlled"):
            qc.while_loop((qc.clbits[0], False), body, [qc.qubits[0]], []).c_if(qc.clbits[0], True)

        with self.assertRaisesRegex(NotImplementedError, r"cannot be classically controlled"):
            qc.if_test((qc.clbits[0], False), body, [qc.qubits[0]], []).c_if(qc.clbits[0], True)

        with self.assertRaisesRegex(NotImplementedError, r"cannot be classically controlled"):
            qc.if_else((qc.clbits[0], False), body, body, [qc.qubits[0]], []).c_if(
                qc.clbits[0], True
            )

    def test_nested_parameters_are_recognised(self):
        """Verify that parameters added inside a control-flow operator get added to the outer
        circuit table."""
        x, y = Parameter("x"), Parameter("y")

        with self.subTest("if/else"):
            body1 = QuantumCircuit(1, 1)
            body1.rx(x, 0)
            body2 = QuantumCircuit(1, 1)
            body2.rx(y, 0)

            main = QuantumCircuit(1, 1)
            main.if_else((main.clbits[0], 0), body1, body2, [0], [0])
            self.assertEqual({x, y}, set(main.parameters))

        with self.subTest("while"):
            body = QuantumCircuit(1, 1)
            body.rx(x, 0)

            main = QuantumCircuit(1, 1)
            main.while_loop((main.clbits[0], 0), body, [0], [0])
            self.assertEqual({x}, set(main.parameters))

        with self.subTest("for"):
            body = QuantumCircuit(1, 1)
            body.rx(x, 0)

            main = QuantumCircuit(1, 1)
            main.for_loop(range(1), None, body, [0], [0])
            self.assertEqual({x}, set(main.parameters))

    def test_nested_parameters_can_be_assigned(self):
        """Verify that parameters added inside a control-flow operator can be assigned by calls to
        the outer circuit."""
        x, y = Parameter("x"), Parameter("y")

        with self.subTest("if/else"):
            body1 = QuantumCircuit(1, 1)
            body1.rx(x, 0)
            body2 = QuantumCircuit(1, 1)
            body2.rx(y, 0)

            test = QuantumCircuit(1, 1)
            test.if_else((test.clbits[0], 0), body1, body2, [0], [0])
            self.assertEqual({x, y}, set(test.parameters))
            assigned = test.assign_parameters({x: math.pi, y: 0.5 * math.pi})
            self.assertEqual(set(), set(assigned.parameters))

            expected = QuantumCircuit(1, 1)
            expected.if_else(
                (expected.clbits[0], 0),
                body1.assign_parameters({x: math.pi}),
                body2.assign_parameters({y: 0.5 * math.pi}),
                [0],
                [0],
            )

            self.assertEqual(assigned, expected)

        with self.subTest("while"):
            body = QuantumCircuit(1, 1)
            body.rx(x, 0)

            test = QuantumCircuit(1, 1)
            test.while_loop((test.clbits[0], 0), body, [0], [0])
            self.assertEqual({x}, set(test.parameters))
            assigned = test.assign_parameters({x: math.pi})
            self.assertEqual(set(), set(assigned.parameters))

            expected = QuantumCircuit(1, 1)
            expected.while_loop(
                (expected.clbits[0], 0),
                body.assign_parameters({x: math.pi}),
                [0],
                [0],
            )

            self.assertEqual(assigned, expected)

        with self.subTest("for"):
            body = QuantumCircuit(1, 1)
            body.rx(x, 0)

            test = QuantumCircuit(1, 1)
            test.for_loop(range(1), None, body, [0], [0])
            self.assertEqual({x}, set(test.parameters))
            assigned = test.assign_parameters({x: math.pi})
            self.assertEqual(set(), set(assigned.parameters))

            expected = QuantumCircuit(1, 1)
            expected.for_loop(
                range(1),
                None,
                body.assign_parameters({x: math.pi}),
                [0],
                [0],
            )

            self.assertEqual(assigned, expected)
