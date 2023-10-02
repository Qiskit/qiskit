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
from qiskit.circuit import Store, Clbit, CircuitError
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
