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
from qiskit.circuit import Store, Clbit, CircuitError, QuantumCircuit, ClassicalRegister
from qiskit.circuit.classical import expr, types


class TestStoreInstruction(QiskitTestCase):
    """Tests of the properties of the ``Store`` instruction itself."""

    def test_happy_path_construction(self):
        lvalue = expr.Var.new("a", types.Bool())
        rvalue = expr.lift(Clbit())
        constructed = Store(lvalue, rvalue)
        self.assertIsInstance(constructed, Store)
        self.assertEqual(constructed.lvalue, lvalue)
        self.assertEqual(constructed.rvalue, rvalue)

    def test_implicit_cast(self):
        lvalue = expr.Var.new("a", types.Bool())
        rvalue = expr.Var.new("b", types.Uint(8))
        constructed = Store(lvalue, rvalue)
        self.assertIsInstance(constructed, Store)
        self.assertEqual(constructed.lvalue, lvalue)
        self.assertEqual(constructed.rvalue, expr.Cast(rvalue, types.Bool(), implicit=True))

    def test_rejects_non_lvalue(self):
        not_an_lvalue = expr.logic_and(
            expr.Var.new("a", types.Bool()), expr.Var.new("b", types.Bool())
        )
        rvalue = expr.lift(False)
        with self.assertRaisesRegex(CircuitError, "not an l-value"):
            Store(not_an_lvalue, rvalue)

    def test_rejects_explicit_cast(self):
        lvalue = expr.Var.new("a", types.Uint(16))
        rvalue = expr.Var.new("b", types.Uint(8))
        with self.assertRaisesRegex(CircuitError, "an explicit cast is required"):
            Store(lvalue, rvalue)

    def test_rejects_dangerous_cast(self):
        lvalue = expr.Var.new("a", types.Uint(8))
        rvalue = expr.Var.new("b", types.Uint(16))
        with self.assertRaisesRegex(CircuitError, "an explicit cast is required.*may be lossy"):
            Store(lvalue, rvalue)

    def test_rejects_c_if(self):
        instruction = Store(expr.Var.new("a", types.Bool()), expr.Var.new("b", types.Bool()))
        with self.assertRaises(NotImplementedError):
            instruction.c_if(Clbit(), False)


class TestStoreCircuit(QiskitTestCase):
    """Tests of the `QuantumCircuit.store` method and appends of `Store`."""

    def test_produces_expected_operation(self):
        a = expr.Var.new("a", types.Bool())
        value = expr.Value(True, types.Bool())

        qc = QuantumCircuit(inputs=[a])
        qc.store(a, value)
        self.assertEqual(qc.data[-1].operation, Store(a, value))

        qc = QuantumCircuit(captures=[a])
        qc.store(a, value)
        self.assertEqual(qc.data[-1].operation, Store(a, value))

        qc = QuantumCircuit(declarations=[(a, expr.lift(False))])
        qc.store(a, value)
        self.assertEqual(qc.data[-1].operation, Store(a, value))

    def test_allows_stores_with_clbits(self):
        clbits = [Clbit(), Clbit()]
        a = expr.Var.new("a", types.Bool())
        qc = QuantumCircuit(clbits, inputs=[a])
        qc.store(clbits[0], True)
        qc.store(expr.Var(clbits[1], types.Bool()), a)
        qc.store(clbits[0], clbits[1])
        qc.store(expr.lift(clbits[0]), expr.lift(clbits[1]))
        qc.store(a, expr.lift(clbits[1]))

        expected = [
            Store(expr.lift(clbits[0]), expr.lift(True)),
            Store(expr.lift(clbits[1]), a),
            Store(expr.lift(clbits[0]), expr.lift(clbits[1])),
            Store(expr.lift(clbits[0]), expr.lift(clbits[1])),
            Store(a, expr.lift(clbits[1])),
        ]
        actual = [instruction.operation for instruction in qc.data]
        self.assertEqual(actual, expected)

    def test_allows_stores_with_cregs(self):
        cregs = [ClassicalRegister(8, "cr1"), ClassicalRegister(8, "cr2")]
        a = expr.Var.new("a", types.Uint(8))
        qc = QuantumCircuit(*cregs, captures=[a])
        qc.store(cregs[0], 0xFF)
        qc.store(expr.Var(cregs[1], types.Uint(8)), a)
        qc.store(cregs[0], cregs[1])
        qc.store(expr.lift(cregs[0]), expr.lift(cregs[1]))
        qc.store(a, cregs[1])

        expected = [
            Store(expr.lift(cregs[0]), expr.lift(0xFF)),
            Store(expr.lift(cregs[1]), a),
            Store(expr.lift(cregs[0]), expr.lift(cregs[1])),
            Store(expr.lift(cregs[0]), expr.lift(cregs[1])),
            Store(a, expr.lift(cregs[1])),
        ]
        actual = [instruction.operation for instruction in qc.data]
        self.assertEqual(actual, expected)

    def test_lifts_values(self):
        a = expr.Var.new("a", types.Bool())
        qc = QuantumCircuit(captures=[a])
        qc.store(a, True)
        self.assertEqual(qc.data[-1].operation, Store(a, expr.lift(True)))

        b = expr.Var.new("b", types.Uint(16))
        qc.add_capture(b)
        qc.store(b, 0xFFFF)
        self.assertEqual(qc.data[-1].operation, Store(b, expr.lift(0xFFFF)))

    def test_rejects_vars_not_in_circuit(self):
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())

        qc = QuantumCircuit()
        with self.assertRaisesRegex(CircuitError, "'a'.*not present"):
            qc.store(expr.Var.new("a", types.Bool()), True)

        # Not the same 'a'
        qc.add_input(a)
        with self.assertRaisesRegex(CircuitError, "'a'.*not present"):
            qc.store(expr.Var.new("a", types.Bool()), True)
        with self.assertRaisesRegex(CircuitError, "'b'.*not present"):
            qc.store(a, b)

    def test_rejects_bits_not_in_circuit(self):
        a = expr.Var.new("a", types.Bool())
        clbit = Clbit()
        qc = QuantumCircuit(captures=[a])
        with self.assertRaisesRegex(CircuitError, "not present"):
            qc.store(clbit, False)
        with self.assertRaisesRegex(CircuitError, "not present"):
            qc.store(clbit, a)
        with self.assertRaisesRegex(CircuitError, "not present"):
            qc.store(a, clbit)

    def test_rejects_cregs_not_in_circuit(self):
        a = expr.Var.new("a", types.Uint(8))
        creg = ClassicalRegister(8, "cr1")
        qc = QuantumCircuit(captures=[a])
        with self.assertRaisesRegex(CircuitError, "not present"):
            qc.store(creg, 0xFF)
        with self.assertRaisesRegex(CircuitError, "not present"):
            qc.store(creg, a)
        with self.assertRaisesRegex(CircuitError, "not present"):
            qc.store(a, creg)

    def test_rejects_non_lvalue(self):
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())
        qc = QuantumCircuit(inputs=[a, b])
        not_an_lvalue = expr.logic_and(a, b)
        with self.assertRaisesRegex(CircuitError, "not an l-value"):
            qc.store(not_an_lvalue, expr.lift(False))

    def test_rejects_explicit_cast(self):
        lvalue = expr.Var.new("a", types.Uint(16))
        rvalue = expr.Var.new("b", types.Uint(8))
        qc = QuantumCircuit(inputs=[lvalue, rvalue])
        with self.assertRaisesRegex(CircuitError, "an explicit cast is required"):
            qc.store(lvalue, rvalue)

    def test_rejects_dangerous_cast(self):
        lvalue = expr.Var.new("a", types.Uint(8))
        rvalue = expr.Var.new("b", types.Uint(16))
        qc = QuantumCircuit(inputs=[lvalue, rvalue])
        with self.assertRaisesRegex(CircuitError, "an explicit cast is required.*may be lossy"):
            qc.store(lvalue, rvalue)

    def test_rejects_c_if(self):
        a = expr.Var.new("a", types.Bool())
        qc = QuantumCircuit([Clbit()], inputs=[a])
        instruction_set = qc.store(a, True)
        with self.assertRaises(NotImplementedError):
            instruction_set.c_if(qc.clbits[0], False)
