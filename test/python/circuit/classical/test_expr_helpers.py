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

import copy
import ddt

from qiskit.circuit import Clbit, ClassicalRegister
from qiskit.circuit.classical import expr, types
from qiskit.test import QiskitTestCase


@ddt.ddt
class TestStructurallyEquivalent(QiskitTestCase):
    @ddt.data(
        expr.lift(Clbit()),
        expr.lift(ClassicalRegister(3, "a")),
        expr.lift(3, types.Uint(2)),
        expr.cast(ClassicalRegister(3, "a"), types.Bool()),
        expr.logic_not(Clbit()),
        expr.bit_and(5, ClassicalRegister(3, "a")),
        expr.logic_and(expr.less(2, ClassicalRegister(3, "a")), expr.lift(Clbit())),
    )
    def test_equivalent_to_self(self, node):
        self.assertTrue(expr.structurally_equivalent(node, node))
        self.assertTrue(expr.structurally_equivalent(node, copy.copy(node)))

    # Not all of the `Binary.Op` opcodes are actually symmetric operations, but the asymmetric ones
    # _definitely_ shouldn't compare equal when flipped!
    @ddt.idata(expr.Binary.Op)
    def test_does_not_compare_symmetrically(self, opcode):
        """The function is specifically not meant to attempt things like flipping the symmetry of
        equality.  We want the function to be simple and predictable to reason about, and allowing
        flipping of even the mathematically symmetric binary operations are not necessarily
        symmetric programmatically; the changed order of operations can have an effect in (say)
        short-circuiting operations, or in external functional calls that modify global state."""
        if opcode in (expr.Binary.Op.LOGIC_AND, expr.Binary.Op.LOGIC_OR):
            left = expr.Value(True, types.Bool())
            right = expr.Var(Clbit(), types.Bool())
        else:
            left = expr.Value(5, types.Uint(3))
            right = expr.Var(ClassicalRegister(3, "a"), types.Uint(3))
        if opcode in (expr.Binary.Op.BIT_AND, expr.Binary.Op.BIT_OR, expr.Binary.Op.BIT_XOR):
            out_type = types.Uint(3)
        else:
            out_type = types.Bool()
        cis = expr.Binary(opcode, left, right, out_type)
        trans = expr.Binary(opcode, right, left, out_type)
        self.assertFalse(expr.structurally_equivalent(cis, trans))
        self.assertFalse(expr.structurally_equivalent(trans, cis))

    def test_key_function_both(self):
        left_clbit = Clbit()
        left_cr = ClassicalRegister(3, "a")
        right_clbit = Clbit()
        right_cr = ClassicalRegister(3, "b")
        self.assertNotEqual(left_clbit, right_clbit)
        self.assertNotEqual(left_cr, right_cr)

        left = expr.logic_not(expr.logic_and(expr.less(5, left_cr), left_clbit))
        right = expr.logic_not(expr.logic_and(expr.less(5, right_cr), right_clbit))

        self.assertFalse(expr.structurally_equivalent(left, right))
        # The only two variables are a `Clbit` and a `ClassicalRegister`, so keying them on their
        # types should give us a suitable comparison.
        self.assertTrue(expr.structurally_equivalent(left, right, type, type))

    def test_key_function_only_one(self):
        left_clbit = Clbit()
        left_cr = ClassicalRegister(3, "a")
        right_clbit = Clbit()
        right_cr = ClassicalRegister(3, "b")
        self.assertNotEqual(left_clbit, right_clbit)
        self.assertNotEqual(left_cr, right_cr)

        left = expr.logic_not(expr.logic_and(expr.less(5, left_cr), left_clbit))
        right = expr.logic_not(expr.logic_and(expr.less(5, right_cr), right_clbit))

        left_to_right = {left_clbit: right_clbit, left_cr: right_cr}.get

        self.assertFalse(expr.structurally_equivalent(left, right))
        self.assertTrue(expr.structurally_equivalent(left, right, left_to_right, None))
        self.assertTrue(expr.structurally_equivalent(right, left, None, left_to_right))

    def test_key_function_can_return_none(self):
        """If the key function returns ``None``, the variable should be used raw as the comparison
        base, _not_ the ``None`` return value."""
        left_bit = Clbit()
        right_bit = Clbit()

        class EqualsEverything:
            def __eq__(self, _other):
                return True

        def not_handled(_):
            return None

        def always_equal(_):
            return EqualsEverything()

        left = expr.logic_and(left_bit, True)
        right = expr.logic_and(right_bit, True)
        self.assertFalse(expr.structurally_equivalent(left, right))
        # If the function erroneously compares the ``None`` outputs, the following call would return
        # ``True`` instead.
        self.assertFalse(expr.structurally_equivalent(left, right, not_handled, not_handled))
        self.assertTrue(expr.structurally_equivalent(left, right, always_equal, always_equal))
