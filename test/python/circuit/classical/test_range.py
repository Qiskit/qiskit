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

"""Test the Range expression class."""

import ddt

from qiskit.circuit.classical import expr, types
from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt.ddt
class TestRange(QiskitTestCase):
    """Test the Range expression class."""

    def test_range_with_uint(self):
        """Test creating a Range with Uint values."""
        start = expr.lift(5, types.Uint(8))
        stop = expr.lift(10, types.Uint(8))
        step = expr.lift(2, types.Uint(8))

        range_expr = expr.Range(start, stop, step)

        self.assertEqual(range_expr.start, start)
        self.assertEqual(range_expr.stop, stop)
        self.assertEqual(range_expr.step, step)
        self.assertEqual(range_expr.type, types.Uint(8))
        self.assertTrue(range_expr.const)

    def test_range_with_different_uint_sizes(self):
        """Test creating a Range with different Uint sizes."""
        start = expr.lift(5, types.Uint(8))
        stop = expr.lift(10, types.Uint(16))
        step = expr.lift(2, types.Uint(32))

        range_expr = expr.Range(start, stop, step)

        self.assertEqual(range_expr.start, start)
        self.assertEqual(range_expr.stop, stop)
        self.assertEqual(range_expr.step, step)
        # Type should be the largest of the inputs (Uint(32))
        self.assertEqual(range_expr.type, types.Uint(32))
        self.assertTrue(range_expr.const)

    def test_range_without_step(self):
        """Test creating a Range without a step value."""
        start = expr.lift(0, types.Uint(8))
        stop = expr.lift(5, types.Uint(8))

        range_expr = expr.Range(start, stop)

        self.assertEqual(range_expr.start, start)
        self.assertEqual(range_expr.stop, stop)
        self.assertIsNone(range_expr.step)
        self.assertEqual(range_expr.type, types.Uint(8))
        self.assertTrue(range_expr.const)

    def test_range_with_float_values(self):
        """Test that creating a Range with float values raises an error."""
        start = expr.lift(5.0, types.Float())
        stop = expr.lift(10.0, types.Float())

        with self.assertRaisesRegex(TypeError, "Range values must be of integer type"):
            expr.Range(start, stop)

    def test_range_with_mixed_types(self):
        """Test that creating a Range with mixed integer and float types raises an error."""
        start = expr.lift(5, types.Uint(8))
        stop = expr.lift(10.0, types.Float())

        with self.assertRaisesRegex(TypeError, "Range values must be of integer type"):
            expr.Range(start, stop)

    def test_range_with_non_constant_values(self):
        """Test creating a Range with non-constant values."""
        from qiskit.circuit import ClassicalRegister

        cr = ClassicalRegister(8, "c")

        start = expr.lift(cr)
        stop = expr.lift(10, types.Uint(8))

        range_expr = expr.Range(start, stop)

        self.assertEqual(range_expr.start, start)
        self.assertEqual(range_expr.stop, stop)
        self.assertIsNone(range_expr.step)
        self.assertEqual(range_expr.type, types.Uint(8))
        self.assertFalse(range_expr.const)

    def test_range_with_invalid_type_specification(self):
        """Test that specifying a non-unsigned integer type raises an error."""
        start = expr.lift(5, types.Uint(8))
        stop = expr.lift(10, types.Uint(8))

        with self.assertRaisesRegex(TypeError, "Range type must be an unsigned integer type"):
            expr.Range(start, stop, ty=types.Float())

    def test_range_with_valid_uint_type_specification(self):
        """Test that specifying a valid Uint type works correctly."""
        start = expr.lift(5, types.Uint(8))
        stop = expr.lift(10, types.Uint(8))

        range_expr = expr.Range(start, stop, ty=types.Uint(16))
        self.assertEqual(range_expr.type, types.Uint(16))
        self.assertEqual(range_expr.start.type, types.Uint(16))
        self.assertEqual(range_expr.stop.type, types.Uint(16))

    def test_range_with_different_input_types(self):
        """Test that values are lifted to the specified type."""
        start = expr.lift(5, types.Uint(8))
        stop = expr.lift(10, types.Uint(16))
        step = expr.lift(2, types.Uint(32))

        range_expr = expr.Range(start, stop, step, ty=types.Uint(64))
        self.assertEqual(range_expr.type, types.Uint(64))
        self.assertEqual(range_expr.start.type, types.Uint(64))
        self.assertEqual(range_expr.stop.type, types.Uint(64))
        self.assertEqual(range_expr.step.type, types.Uint(64))

    def test_range_with_step_lifted_to_type(self):
        """Test that step value is lifted to the specified type."""
        start = expr.lift(5, types.Uint(8))
        stop = expr.lift(10, types.Uint(8))

        range_expr = expr.Range(start, stop, step=expr.lift(2, types.Uint(16)), ty=types.Uint(32))
        self.assertEqual(range_expr.type, types.Uint(32))
        self.assertEqual(range_expr.start.type, types.Uint(32))
        self.assertEqual(range_expr.stop.type, types.Uint(32))
        self.assertEqual(range_expr.step.type, types.Uint(32))

    def test_range_with_valid_type_casting(self):
        """Test that values are properly cast to the specified type."""
        # Create expressions with different types
        start = expr.lift(5, types.Uint(8))
        stop = expr.lift(10, types.Uint(16))
        step = expr.lift(2, types.Uint(32))

        # Cast all to Uint(64)
        range_expr = expr.Range(start, stop, step, ty=types.Uint(64))
        
        # Verify all expressions were cast to Uint(64)
        self.assertEqual(range_expr.type, types.Uint(64))
        self.assertEqual(range_expr.start.type, types.Uint(64))
        self.assertEqual(range_expr.stop.type, types.Uint(64))
        self.assertEqual(range_expr.step.type, types.Uint(64))

        # Verify the expressions are Cast nodes
        self.assertIsInstance(range_expr.start, expr.Cast)
        self.assertIsInstance(range_expr.stop, expr.Cast)
        self.assertIsInstance(range_expr.step, expr.Cast)

        # Verify the casts are explicit (not implicit)
        self.assertFalse(range_expr.start.implicit)
        self.assertFalse(range_expr.stop.implicit)
        self.assertFalse(range_expr.step.implicit)

    def test_range_with_invalid_type_casting(self):
        """Test that invalid type casts raise appropriate errors."""
        # Create expressions with different types
        start = expr.lift(5, types.Uint(8))
        stop = expr.lift(10, types.Uint(16))
        step = expr.lift(2, types.Uint(32))

        # Try to cast to a smaller type (should fail)
        with self.assertRaisesRegex(TypeError, "Cannot cast range values to the specified type"):
            expr.Range(start, stop, step, ty=types.Uint(4))

        # Try to cast to a non-uint type (should fail)
        with self.assertRaisesRegex(TypeError, "Range type must be an unsigned integer type"):
            expr.Range(start, stop, step, ty=types.Float())

    def test_range_with_mixed_type_casting(self):
        """Test casting with mixed constant and non-constant expressions."""
        from qiskit.circuit import ClassicalRegister

        cr = ClassicalRegister(8, "c")
        start = expr.lift(cr)  # Non-constant
        stop = expr.lift(10, types.Uint(8))  # Constant
        step = expr.lift(2, types.Uint(16))  # Constant

        # Cast all to Uint(32)
        range_expr = expr.Range(start, stop, step, ty=types.Uint(32))
        
        # Verify all expressions were cast to Uint(32)
        self.assertEqual(range_expr.type, types.Uint(32))
        self.assertEqual(range_expr.start.type, types.Uint(32))
        self.assertEqual(range_expr.stop.type, types.Uint(32))
        self.assertEqual(range_expr.step.type, types.Uint(32))

        # Verify constant flag is preserved
        self.assertFalse(range_expr.start.constant)  # Non-constant
        self.assertTrue(range_expr.stop.constant)    # Constant
        self.assertTrue(range_expr.step.constant)    # Constant
