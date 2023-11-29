# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from qiskit.test import QiskitTestCase
from qiskit.circuit import QuantumCircuit, CircuitError, Clbit, ClassicalRegister
from qiskit.circuit.classical import expr, types


class TestCircuitVars(QiskitTestCase):
    """Tests for variable-manipulation routines on circuits.  More specific functionality is likely
    tested in the suites of the specific methods."""

    def test_initialise_inputs(self):
        vars_ = [expr.Var.new("a", types.Bool()), expr.Var.new("b", types.Uint(16))]
        qc = QuantumCircuit(inputs=vars_)
        self.assertEqual(set(vars_), set(qc.iter_vars()))
        self.assertEqual(qc.num_vars, len(vars_))
        self.assertEqual(qc.num_input_vars, len(vars_))
        self.assertEqual(qc.num_captured_vars, 0)
        self.assertEqual(qc.num_declared_vars, 0)

    def test_initialise_captures(self):
        vars_ = [expr.Var.new("a", types.Bool()), expr.Var.new("b", types.Uint(16))]
        qc = QuantumCircuit(captures=vars_)
        self.assertEqual(set(vars_), set(qc.iter_vars()))
        self.assertEqual(qc.num_vars, len(vars_))
        self.assertEqual(qc.num_input_vars, 0)
        self.assertEqual(qc.num_captured_vars, len(vars_))
        self.assertEqual(qc.num_declared_vars, 0)

    def test_initialise_declarations_iterable(self):
        vars_ = [
            (expr.Var.new("a", types.Bool()), expr.lift(True)),
            (expr.Var.new("b", types.Uint(16)), expr.lift(0xFFFF)),
        ]
        qc = QuantumCircuit(declarations=vars_)

        self.assertEqual({var for var, _initialiser in vars_}, set(qc.iter_vars()))
        self.assertEqual(qc.num_vars, len(vars_))
        self.assertEqual(qc.num_input_vars, 0)
        self.assertEqual(qc.num_captured_vars, 0)
        self.assertEqual(qc.num_declared_vars, len(vars_))
        operations = [
            (instruction.operation.name, instruction.operation.lvalue, instruction.operation.rvalue)
            for instruction in qc.data
        ]
        self.assertEqual(operations, [("store", lvalue, rvalue) for lvalue, rvalue in vars_])

    def test_initialise_declarations_mapping(self):
        # Dictionary iteration order is guaranteed to be insertion order.
        vars_ = {
            expr.Var.new("a", types.Bool()): expr.lift(True),
            expr.Var.new("b", types.Uint(16)): expr.lift(0xFFFF),
        }
        qc = QuantumCircuit(declarations=vars_)

        self.assertEqual(set(vars_), set(qc.iter_vars()))
        operations = [
            (instruction.operation.name, instruction.operation.lvalue, instruction.operation.rvalue)
            for instruction in qc.data
        ]
        self.assertEqual(
            operations, [("store", lvalue, rvalue) for lvalue, rvalue in vars_.items()]
        )

    def test_initialise_declarations_dependencies(self):
        """Test that the cirucit initialiser can take in declarations with dependencies between
        them, provided they're specified in a suitable order."""
        a = expr.Var.new("a", types.Bool())
        vars_ = [
            (a, expr.lift(True)),
            (expr.Var.new("b", types.Bool()), a),
        ]
        qc = QuantumCircuit(declarations=vars_)

        self.assertEqual({var for var, _initialiser in vars_}, set(qc.iter_vars()))
        operations = [
            (instruction.operation.name, instruction.operation.lvalue, instruction.operation.rvalue)
            for instruction in qc.data
        ]
        self.assertEqual(operations, [("store", lvalue, rvalue) for lvalue, rvalue in vars_])

    def test_initialise_inputs_declarations(self):
        a = expr.Var.new("a", types.Uint(16))
        b = expr.Var.new("b", types.Uint(16))
        b_init = expr.bit_and(a, 0xFFFF)
        qc = QuantumCircuit(inputs=[a], declarations={b: b_init})

        self.assertEqual({a}, set(qc.iter_input_vars()))
        self.assertEqual({b}, set(qc.iter_declared_vars()))
        self.assertEqual({a, b}, set(qc.iter_vars()))
        self.assertEqual(qc.num_vars, 2)
        self.assertEqual(qc.num_input_vars, 1)
        self.assertEqual(qc.num_captured_vars, 0)
        self.assertEqual(qc.num_declared_vars, 1)
        operations = [
            (instruction.operation.name, instruction.operation.lvalue, instruction.operation.rvalue)
            for instruction in qc.data
        ]
        self.assertEqual(operations, [("store", b, b_init)])

    def test_initialise_captures_declarations(self):
        a = expr.Var.new("a", types.Uint(16))
        b = expr.Var.new("b", types.Uint(16))
        b_init = expr.bit_and(a, 0xFFFF)
        qc = QuantumCircuit(captures=[a], declarations={b: b_init})

        self.assertEqual({a}, set(qc.iter_captured_vars()))
        self.assertEqual({b}, set(qc.iter_declared_vars()))
        self.assertEqual({a, b}, set(qc.iter_vars()))
        self.assertEqual(qc.num_vars, 2)
        self.assertEqual(qc.num_input_vars, 0)
        self.assertEqual(qc.num_captured_vars, 1)
        self.assertEqual(qc.num_declared_vars, 1)
        operations = [
            (instruction.operation.name, instruction.operation.lvalue, instruction.operation.rvalue)
            for instruction in qc.data
        ]
        self.assertEqual(operations, [("store", b, b_init)])

    def test_add_uninitialized_var(self):
        a = expr.Var.new("a", types.Bool())
        qc = QuantumCircuit()
        qc.add_uninitialized_var(a)
        self.assertEqual({a}, set(qc.iter_vars()))
        self.assertEqual([], list(qc.data))

    def test_add_var_returns_good_var(self):
        qc = QuantumCircuit()
        a = qc.add_var("a", expr.lift(True))
        self.assertEqual(a.name, "a")
        self.assertEqual(a.type, types.Bool())

        b = qc.add_var("b", expr.Value(0xFF, types.Uint(8)))
        self.assertEqual(b.name, "b")
        self.assertEqual(b.type, types.Uint(8))

    def test_add_var_returns_input(self):
        """Test that the `Var` returned by `add_var` is the same as the input if `Var`."""
        a = expr.Var.new("a", types.Bool())
        qc = QuantumCircuit()
        a_other = qc.add_var(a, expr.lift(True))
        self.assertIs(a, a_other)

    def test_add_input_returns_good_var(self):
        qc = QuantumCircuit()
        a = qc.add_input("a", types.Bool())
        self.assertEqual(a.name, "a")
        self.assertEqual(a.type, types.Bool())

        b = qc.add_input("b", types.Uint(8))
        self.assertEqual(b.name, "b")
        self.assertEqual(b.type, types.Uint(8))

    def test_add_input_returns_input(self):
        """Test that the `Var` returned by `add_input` is the same as the input if `Var`."""
        a = expr.Var.new("a", types.Bool())
        qc = QuantumCircuit()
        a_other = qc.add_input(a)
        self.assertIs(a, a_other)

    def test_cannot_have_both_inputs_and_captures(self):
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())

        with self.assertRaisesRegex(CircuitError, "circuits with input.*cannot be closures"):
            QuantumCircuit(inputs=[a], captures=[b])

        qc = QuantumCircuit(inputs=[a])
        with self.assertRaisesRegex(CircuitError, "circuits with input.*cannot be closures"):
            qc.add_capture(b)

        qc = QuantumCircuit(captures=[a])
        with self.assertRaisesRegex(CircuitError, "circuits to be enclosed.*cannot have input"):
            qc.add_input(b)

    def test_cannot_add_cyclic_declaration(self):
        a = expr.Var.new("a", types.Bool())
        with self.assertRaisesRegex(CircuitError, "not present in this circuit"):
            QuantumCircuit(declarations=[(a, a)])

        qc = QuantumCircuit()
        with self.assertRaisesRegex(CircuitError, "not present in this circuit"):
            qc.add_var(a, a)

    def test_initialise_inputs_equal_to_add_input(self):
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(16))

        qc_init = QuantumCircuit(inputs=[a, b])
        qc_manual = QuantumCircuit()
        qc_manual.add_input(a)
        qc_manual.add_input(b)
        self.assertEqual(list(qc_init.iter_vars()), list(qc_manual.iter_vars()))

        qc_manual = QuantumCircuit()
        a = qc_manual.add_input("a", types.Bool())
        b = qc_manual.add_input("b", types.Uint(16))
        qc_init = QuantumCircuit(inputs=[a, b])
        self.assertEqual(list(qc_init.iter_vars()), list(qc_manual.iter_vars()))

    def test_initialise_captures_equal_to_add_capture(self):
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(16))

        qc_init = QuantumCircuit(captures=[a, b])
        qc_manual = QuantumCircuit()
        qc_manual.add_capture(a)
        qc_manual.add_capture(b)
        self.assertEqual(list(qc_init.iter_vars()), list(qc_manual.iter_vars()))

    def test_initialise_declarations_equal_to_add_var(self):
        a = expr.Var.new("a", types.Bool())
        a_init = expr.lift(False)
        b = expr.Var.new("b", types.Uint(16))
        b_init = expr.lift(0xFFFF)

        qc_init = QuantumCircuit(declarations=[(a, a_init), (b, b_init)])
        qc_manual = QuantumCircuit()
        qc_manual.add_var(a, a_init)
        qc_manual.add_var(b, b_init)
        self.assertEqual(list(qc_init.iter_vars()), list(qc_manual.iter_vars()))
        self.assertEqual(qc_init.data, qc_manual.data)

        qc_manual = QuantumCircuit()
        a = qc_manual.add_var("a", a_init)
        b = qc_manual.add_var("b", b_init)
        qc_init = QuantumCircuit(declarations=[(a, a_init), (b, b_init)])
        self.assertEqual(list(qc_init.iter_vars()), list(qc_manual.iter_vars()))
        self.assertEqual(qc_init.data, qc_manual.data)

    def test_cannot_shadow_vars(self):
        """Test that exact duplicate ``Var`` nodes within different combinations of the inputs are
        detected and rejected."""
        a = expr.Var.new("a", types.Bool())
        a_init = expr.lift(True)
        with self.assertRaisesRegex(CircuitError, "already present"):
            QuantumCircuit(inputs=[a, a])
        with self.assertRaisesRegex(CircuitError, "already present"):
            QuantumCircuit(captures=[a, a])
        with self.assertRaisesRegex(CircuitError, "already present"):
            QuantumCircuit(declarations=[(a, a_init), (a, a_init)])
        with self.assertRaisesRegex(CircuitError, "already present"):
            QuantumCircuit(inputs=[a], declarations=[(a, a_init)])
        with self.assertRaisesRegex(CircuitError, "already present"):
            QuantumCircuit(captures=[a], declarations=[(a, a_init)])

    def test_cannot_shadow_names(self):
        """Test that exact duplicate ``Var`` nodes within different combinations of the inputs are
        detected and rejected."""
        a_bool1 = expr.Var.new("a", types.Bool())
        a_bool2 = expr.Var.new("a", types.Bool())
        a_uint = expr.Var.new("a", types.Uint(16))
        a_bool_init = expr.lift(True)
        a_uint_init = expr.lift(0xFFFF)

        tests = [
            ((a_bool1, a_bool_init), (a_bool2, a_bool_init)),
            ((a_bool1, a_bool_init), (a_uint, a_uint_init)),
        ]
        for (left, left_init), (right, right_init) in tests:
            with self.assertRaisesRegex(CircuitError, "its name shadows"):
                QuantumCircuit(inputs=(left, right))
            with self.assertRaisesRegex(CircuitError, "its name shadows"):
                QuantumCircuit(captures=(left, right))
            with self.assertRaisesRegex(CircuitError, "its name shadows"):
                QuantumCircuit(declarations=[(left, left_init), (right, right_init)])
            with self.assertRaisesRegex(CircuitError, "its name shadows"):
                QuantumCircuit(inputs=[left], declarations=[(right, right_init)])
            with self.assertRaisesRegex(CircuitError, "its name shadows"):
                QuantumCircuit(captures=[left], declarations=[(right, right_init)])

            qc = QuantumCircuit(inputs=[left])
            with self.assertRaisesRegex(CircuitError, "its name shadows"):
                qc.add_input(right)
            qc = QuantumCircuit(inputs=[left])
            with self.assertRaisesRegex(CircuitError, "its name shadows"):
                qc.add_var(right, right_init)

            qc = QuantumCircuit(captures=[left])
            with self.assertRaisesRegex(CircuitError, "its name shadows"):
                qc.add_capture(right)
            qc = QuantumCircuit(captures=[left])
            with self.assertRaisesRegex(CircuitError, "its name shadows"):
                qc.add_var(right, right_init)

            qc = QuantumCircuit(inputs=[left])
            with self.assertRaisesRegex(CircuitError, "its name shadows"):
                qc.add_var(right, right_init)

        qc = QuantumCircuit()
        qc.add_var("a", expr.lift(True))
        with self.assertRaisesRegex(CircuitError, "its name shadows"):
            qc.add_var("a", expr.lift(True))
        with self.assertRaisesRegex(CircuitError, "its name shadows"):
            qc.add_var("a", expr.lift(0xFF))

    def test_cannot_add_vars_wrapping_clbits(self):
        a = expr.Var(Clbit(), types.Bool())
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            QuantumCircuit(inputs=[a])
        qc = QuantumCircuit()
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            qc.add_input(a)
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            QuantumCircuit(captures=[a])
        qc = QuantumCircuit()
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            qc.add_capture(a)
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            QuantumCircuit(declarations=[(a, expr.lift(True))])
        qc = QuantumCircuit()
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            qc.add_var(a, expr.lift(True))

    def test_cannot_add_vars_wrapping_cregs(self):
        a = expr.Var(ClassicalRegister(8, "cr"), types.Uint(8))
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            QuantumCircuit(inputs=[a])
        qc = QuantumCircuit()
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            qc.add_input(a)
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            QuantumCircuit(captures=[a])
        qc = QuantumCircuit()
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            qc.add_capture(a)
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            QuantumCircuit(declarations=[(a, expr.lift(0xFF))])
        qc = QuantumCircuit()
        with self.assertRaisesRegex(CircuitError, "cannot add variables that wrap"):
            qc.add_var(a, expr.lift(0xFF))

    def test_get_var_success(self):
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))

        qc = QuantumCircuit(inputs=[a], declarations={b: expr.Value(0xFF, types.Uint(8))})
        self.assertIs(qc.get_var("a"), a)
        self.assertIs(qc.get_var("b"), b)

        qc = QuantumCircuit(captures=[a, b])
        self.assertIs(qc.get_var("a"), a)
        self.assertIs(qc.get_var("b"), b)

        qc = QuantumCircuit(declarations={a: expr.lift(True), b: expr.Value(0xFF, types.Uint(8))})
        self.assertIs(qc.get_var("a"), a)
        self.assertIs(qc.get_var("b"), b)

    def test_get_var_missing(self):
        qc = QuantumCircuit()
        with self.assertRaises(KeyError):
            qc.get_var("a")

        a = expr.Var.new("a", types.Bool())
        qc.add_input(a)
        with self.assertRaises(KeyError):
            qc.get_var("b")

    def test_get_var_default(self):
        qc = QuantumCircuit()
        self.assertIs(qc.get_var("a", None), None)

        missing = "default"
        a = expr.Var.new("a", types.Bool())
        qc.add_input(a)
        self.assertIs(qc.get_var("b", missing), missing)
        self.assertIs(qc.get_var("b", a), a)

    def test_has_var(self):
        a = expr.Var.new("a", types.Bool())
        self.assertFalse(QuantumCircuit().has_var("a"))
        self.assertTrue(QuantumCircuit(inputs=[a]).has_var("a"))
        self.assertTrue(QuantumCircuit(captures=[a]).has_var("a"))
        self.assertTrue(QuantumCircuit(declarations={a: expr.lift(True)}).has_var("a"))
        self.assertTrue(QuantumCircuit(inputs=[a]).has_var(a))
        self.assertTrue(QuantumCircuit(captures=[a]).has_var(a))
        self.assertTrue(QuantumCircuit(declarations={a: expr.lift(True)}).has_var(a))

        # When giving an `Var`, the match must be exact, not just the name.
        self.assertFalse(QuantumCircuit(inputs=[a]).has_var(expr.Var.new("a", types.Uint(8))))
        self.assertFalse(QuantumCircuit(inputs=[a]).has_var(expr.Var.new("a", types.Bool())))
