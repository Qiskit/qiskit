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

from qiskit.circuit.classical import types
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestTypesOrdering(QiskitTestCase):
    def test_order(self):
        self.assertIs(types.order(types.Uint(8), types.Uint(16)), types.Ordering.LESS)
        self.assertIs(types.order(types.Uint(16), types.Uint(8)), types.Ordering.GREATER)
        self.assertIs(types.order(types.Uint(8), types.Uint(8)), types.Ordering.EQUAL)

        self.assertIs(types.order(types.Bool(), types.Bool()), types.Ordering.EQUAL)

        self.assertIs(types.order(types.Bool(), types.Uint(8)), types.Ordering.NONE)
        self.assertIs(types.order(types.Uint(8), types.Bool()), types.Ordering.NONE)

    def test_is_subtype(self):
        self.assertTrue(types.is_subtype(types.Uint(8), types.Uint(16)))
        self.assertFalse(types.is_subtype(types.Uint(16), types.Uint(8)))
        self.assertTrue(types.is_subtype(types.Uint(8), types.Uint(8)))
        self.assertFalse(types.is_subtype(types.Uint(8), types.Uint(8), strict=True))

        self.assertTrue(types.is_subtype(types.Bool(), types.Bool()))
        self.assertFalse(types.is_subtype(types.Bool(), types.Bool(), strict=True))

        self.assertFalse(types.is_subtype(types.Bool(), types.Uint(8)))
        self.assertFalse(types.is_subtype(types.Uint(8), types.Bool()))

    def test_is_supertype(self):
        self.assertFalse(types.is_supertype(types.Uint(8), types.Uint(16)))
        self.assertTrue(types.is_supertype(types.Uint(16), types.Uint(8)))
        self.assertTrue(types.is_supertype(types.Uint(8), types.Uint(8)))
        self.assertFalse(types.is_supertype(types.Uint(8), types.Uint(8), strict=True))

        self.assertTrue(types.is_supertype(types.Bool(), types.Bool()))
        self.assertFalse(types.is_supertype(types.Bool(), types.Bool(), strict=True))

        self.assertFalse(types.is_supertype(types.Bool(), types.Uint(8)))
        self.assertFalse(types.is_supertype(types.Uint(8), types.Bool()))

    def test_greater(self):
        self.assertEqual(types.greater(types.Uint(16), types.Uint(8)), types.Uint(16))
        self.assertEqual(types.greater(types.Uint(8), types.Uint(16)), types.Uint(16))
        self.assertEqual(types.greater(types.Uint(8), types.Uint(8)), types.Uint(8))
        self.assertEqual(types.greater(types.Bool(), types.Bool()), types.Bool())
        with self.assertRaisesRegex(TypeError, "no ordering"):
            types.greater(types.Bool(), types.Uint(8))


class TestTypesCastKind(QiskitTestCase):
    def test_basic_examples(self):
        """This is used extensively throughout the expression construction functions, but since it
        is public API, it should have some direct unit tests as well."""
        self.assertIs(types.cast_kind(types.Bool(), types.Bool()), types.CastKind.EQUAL)
        self.assertIs(types.cast_kind(types.Uint(8), types.Bool()), types.CastKind.IMPLICIT)
        self.assertIs(types.cast_kind(types.Bool(), types.Uint(8)), types.CastKind.LOSSLESS)
        self.assertIs(types.cast_kind(types.Uint(16), types.Uint(8)), types.CastKind.DANGEROUS)
