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
        self.assertTrue(range_expr.constant)

    def test_range_with_float(self):
        """Test creating a Range with Float values."""
        start = expr.lift(1.5, types.Float())
        stop = expr.lift(5.0, types.Float())
        step = expr.lift(0.5, types.Float())

        range_expr = expr.Range(start, stop, step)

        self.assertEqual(range_expr.start, start)
        self.assertEqual(range_expr.stop, stop)
        self.assertEqual(range_expr.step, step)
        self.assertEqual(range_expr.type, types.Float())
        self.assertTrue(range_expr.constant)

    def test_range_without_step(self):
        """Test creating a Range without a step value."""
        start = expr.lift(0, types.Uint(8))
        stop = expr.lift(5, types.Uint(8))

        range_expr = expr.Range(start, stop)

        self.assertEqual(range_expr.start, start)
        self.assertEqual(range_expr.stop, stop)
        self.assertIsNone(range_expr.step)
        self.assertEqual(range_expr.type, types.Uint(8))
        self.assertTrue(range_expr.constant)

    def test_range_with_mixed_types(self):
        """Test that creating a Range with mixed types raises an error."""
        start = expr.lift(5, types.Uint(8))
        stop = expr.lift(10.0, types.Float())

        with self.assertRaisesRegex(TypeError, "invalid types for range"):
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
        self.assertFalse(range_expr.constant)
