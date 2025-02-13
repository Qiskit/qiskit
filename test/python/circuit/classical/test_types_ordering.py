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
        self.assertIs(types.order(types.Uint(8, const=True), types.Uint(16)), types.Ordering.LESS)
        self.assertIs(types.order(types.Uint(8), types.Uint(16, const=True)), types.Ordering.NONE)
        self.assertIs(
            types.order(types.Uint(8, const=True), types.Uint(16, const=True)), types.Ordering.LESS
        )

        self.assertIs(types.order(types.Uint(16), types.Uint(8)), types.Ordering.GREATER)
        self.assertIs(
            types.order(types.Uint(16), types.Uint(8, const=True)), types.Ordering.GREATER
        )
        self.assertIs(types.order(types.Uint(16, const=True), types.Uint(8)), types.Ordering.NONE)
        self.assertIs(
            types.order(types.Uint(16, const=True), types.Uint(8, const=True)),
            types.Ordering.GREATER,
        )

        self.assertIs(types.order(types.Uint(8), types.Uint(8)), types.Ordering.EQUAL)
        self.assertIs(types.order(types.Uint(8, const=True), types.Uint(8)), types.Ordering.LESS)
        self.assertIs(types.order(types.Uint(8), types.Uint(8, const=True)), types.Ordering.GREATER)
        self.assertIs(
            types.order(types.Uint(8, const=True), types.Uint(8, const=True)), types.Ordering.EQUAL
        )

        self.assertIs(types.order(types.Bool(), types.Bool()), types.Ordering.EQUAL)
        self.assertIs(types.order(types.Bool(const=True), types.Bool()), types.Ordering.LESS)
        self.assertIs(types.order(types.Bool(), types.Bool(const=True)), types.Ordering.GREATER)
        self.assertIs(
            types.order(types.Bool(const=True), types.Bool(const=True)), types.Ordering.EQUAL
        )

        self.assertIs(types.order(types.Float(), types.Float()), types.Ordering.EQUAL)
        self.assertIs(types.order(types.Float(const=True), types.Float()), types.Ordering.LESS)
        self.assertIs(types.order(types.Float(), types.Float(const=True)), types.Ordering.GREATER)
        self.assertIs(
            types.order(types.Float(const=True), types.Float(const=True)), types.Ordering.EQUAL
        )

        self.assertIs(types.order(types.Duration(), types.Duration()), types.Ordering.EQUAL)
        self.assertIs(types.order(types.Duration(), types.Stretch()), types.Ordering.LESS)
        self.assertIs(types.order(types.Stretch(), types.Duration()), types.Ordering.GREATER)
        self.assertIs(types.order(types.Stretch(), types.Stretch()), types.Ordering.EQUAL)

        self.assertIs(types.order(types.Bool(), types.Uint(8)), types.Ordering.NONE)
        self.assertIs(types.order(types.Bool(), types.Float()), types.Ordering.NONE)
        self.assertIs(types.order(types.Bool(), types.Duration()), types.Ordering.NONE)
        self.assertIs(types.order(types.Bool(), types.Stretch()), types.Ordering.NONE)
        self.assertIs(types.order(types.Uint(8), types.Bool()), types.Ordering.NONE)
        self.assertIs(types.order(types.Uint(8), types.Float()), types.Ordering.NONE)
        self.assertIs(types.order(types.Uint(8), types.Duration()), types.Ordering.NONE)
        self.assertIs(types.order(types.Uint(8), types.Stretch()), types.Ordering.NONE)
        self.assertIs(types.order(types.Float(), types.Uint(8)), types.Ordering.NONE)
        self.assertIs(types.order(types.Float(), types.Bool()), types.Ordering.NONE)
        self.assertIs(types.order(types.Float(), types.Duration()), types.Ordering.NONE)
        self.assertIs(types.order(types.Float(), types.Stretch()), types.Ordering.NONE)
        self.assertIs(types.order(types.Duration(), types.Bool()), types.Ordering.NONE)
        self.assertIs(types.order(types.Duration(), types.Uint(8)), types.Ordering.NONE)
        self.assertIs(types.order(types.Duration(), types.Float()), types.Ordering.NONE)
        self.assertIs(types.order(types.Stretch(), types.Bool()), types.Ordering.NONE)
        self.assertIs(types.order(types.Stretch(), types.Uint(8)), types.Ordering.NONE)
        self.assertIs(types.order(types.Stretch(), types.Float()), types.Ordering.NONE)

    def test_is_subtype(self):
        self.assertTrue(types.is_subtype(types.Uint(8), types.Uint(16)))
        self.assertTrue(types.is_subtype(types.Uint(8, const=True), types.Uint(16)))
        self.assertFalse(types.is_subtype(types.Uint(16), types.Uint(8)))
        self.assertFalse(types.is_subtype(types.Uint(16, const=True), types.Uint(8)))
        self.assertFalse(types.is_subtype(types.Uint(16), types.Uint(8, const=True)))
        self.assertTrue(types.is_subtype(types.Uint(8), types.Uint(8)))
        self.assertFalse(types.is_subtype(types.Uint(8), types.Uint(8), strict=True))
        self.assertTrue(types.is_subtype(types.Uint(8, const=True), types.Uint(8), strict=True))

        self.assertTrue(types.is_subtype(types.Bool(), types.Bool()))
        self.assertFalse(types.is_subtype(types.Bool(), types.Bool(), strict=True))
        self.assertTrue(types.is_subtype(types.Bool(const=True), types.Bool(), strict=True))

        self.assertTrue(types.is_subtype(types.Float(), types.Float()))
        self.assertFalse(types.is_subtype(types.Float(), types.Float(), strict=True))
        self.assertTrue(types.is_subtype(types.Float(const=True), types.Float(), strict=True))

        self.assertTrue(types.is_subtype(types.Duration(), types.Duration()))
        self.assertFalse(types.is_subtype(types.Duration(), types.Duration(), strict=True))
        self.assertTrue(types.is_subtype(types.Duration(), types.Stretch()))
        self.assertTrue(types.is_subtype(types.Duration(), types.Stretch(), strict=True))

        self.assertTrue(types.is_subtype(types.Stretch(), types.Stretch()))
        self.assertFalse(types.is_subtype(types.Stretch(), types.Stretch(), strict=True))

        self.assertFalse(types.is_subtype(types.Bool(), types.Uint(8)))
        self.assertFalse(types.is_subtype(types.Bool(), types.Float()))
        self.assertFalse(types.is_subtype(types.Bool(), types.Duration()))
        self.assertFalse(types.is_subtype(types.Bool(), types.Stretch()))
        self.assertFalse(types.is_subtype(types.Uint(8), types.Bool()))
        self.assertFalse(types.is_subtype(types.Uint(8), types.Float()))
        self.assertFalse(types.is_subtype(types.Uint(8), types.Duration()))
        self.assertFalse(types.is_subtype(types.Uint(8), types.Stretch()))
        self.assertFalse(types.is_subtype(types.Float(), types.Uint(8)))
        self.assertFalse(types.is_subtype(types.Float(), types.Bool()))
        self.assertFalse(types.is_subtype(types.Float(), types.Duration()))
        self.assertFalse(types.is_subtype(types.Float(), types.Stretch()))
        self.assertFalse(types.is_subtype(types.Stretch(), types.Bool()))
        self.assertFalse(types.is_subtype(types.Stretch(), types.Uint(8)))
        self.assertFalse(types.is_subtype(types.Stretch(), types.Float()))
        self.assertFalse(types.is_subtype(types.Stretch(), types.Duration()))
        self.assertFalse(types.is_subtype(types.Duration(), types.Bool()))
        self.assertFalse(types.is_subtype(types.Duration(), types.Uint(8)))
        self.assertFalse(types.is_subtype(types.Duration(), types.Float()))

    def test_is_supertype(self):
        self.assertFalse(types.is_supertype(types.Uint(8), types.Uint(16)))
        self.assertFalse(types.is_supertype(types.Uint(8, const=True), types.Uint(16)))
        self.assertTrue(types.is_supertype(types.Uint(16), types.Uint(8)))
        self.assertTrue(types.is_supertype(types.Uint(16), types.Uint(8, const=True)))
        self.assertTrue(types.is_supertype(types.Uint(16, const=True), types.Uint(8, const=True)))
        self.assertFalse(types.is_supertype(types.Uint(16, const=True), types.Uint(8)))
        self.assertTrue(types.is_supertype(types.Uint(8), types.Uint(8)))
        self.assertFalse(types.is_supertype(types.Uint(8), types.Uint(8), strict=True))
        self.assertTrue(types.is_supertype(types.Uint(8), types.Uint(8, const=True), strict=True))

        self.assertTrue(types.is_supertype(types.Bool(), types.Bool()))
        self.assertFalse(types.is_supertype(types.Bool(), types.Bool(), strict=True))
        self.assertTrue(types.is_supertype(types.Bool(), types.Bool(const=True), strict=True))

        self.assertTrue(types.is_supertype(types.Float(), types.Float()))
        self.assertFalse(types.is_supertype(types.Float(), types.Float(), strict=True))
        self.assertTrue(types.is_supertype(types.Float(), types.Float(const=True), strict=True))

        self.assertTrue(types.is_supertype(types.Duration(), types.Duration()))
        self.assertFalse(types.is_supertype(types.Duration(), types.Duration(), strict=True))

        self.assertTrue(types.is_supertype(types.Stretch(), types.Stretch()))
        self.assertFalse(types.is_supertype(types.Stretch(), types.Stretch(), strict=True))
        self.assertTrue(types.is_supertype(types.Stretch(), types.Duration()))
        self.assertTrue(types.is_supertype(types.Stretch(), types.Duration(), strict=True))

        self.assertFalse(types.is_supertype(types.Bool(), types.Uint(8)))
        self.assertFalse(types.is_supertype(types.Bool(), types.Float()))
        self.assertFalse(types.is_supertype(types.Bool(), types.Duration()))
        self.assertFalse(types.is_supertype(types.Bool(), types.Stretch()))
        self.assertFalse(types.is_supertype(types.Uint(8), types.Bool()))
        self.assertFalse(types.is_supertype(types.Uint(8), types.Float()))
        self.assertFalse(types.is_supertype(types.Uint(8), types.Duration()))
        self.assertFalse(types.is_supertype(types.Uint(8), types.Stretch()))
        self.assertFalse(types.is_supertype(types.Float(), types.Uint(8)))
        self.assertFalse(types.is_supertype(types.Float(), types.Bool()))
        self.assertFalse(types.is_supertype(types.Float(), types.Duration()))
        self.assertFalse(types.is_supertype(types.Float(), types.Stretch()))
        self.assertFalse(types.is_supertype(types.Stretch(), types.Bool()))
        self.assertFalse(types.is_supertype(types.Stretch(), types.Uint(8)))
        self.assertFalse(types.is_supertype(types.Stretch(), types.Float()))
        self.assertFalse(types.is_supertype(types.Duration(), types.Bool()))
        self.assertFalse(types.is_supertype(types.Duration(), types.Uint(8)))
        self.assertFalse(types.is_supertype(types.Duration(), types.Float()))
        self.assertFalse(types.is_supertype(types.Duration(), types.Stretch()))

    def test_greater(self):
        self.assertEqual(types.greater(types.Uint(16), types.Uint(8)), types.Uint(16))
        self.assertEqual(types.greater(types.Uint(16), types.Uint(8, const=True)), types.Uint(16))
        self.assertEqual(types.greater(types.Uint(8), types.Uint(16)), types.Uint(16))
        self.assertEqual(types.greater(types.Uint(8, const=True), types.Uint(16)), types.Uint(16))
        self.assertEqual(types.greater(types.Uint(8), types.Uint(8)), types.Uint(8))
        self.assertEqual(types.greater(types.Uint(8), types.Uint(8, const=True)), types.Uint(8))
        self.assertEqual(types.greater(types.Uint(8, const=True), types.Uint(8)), types.Uint(8))
        self.assertEqual(
            types.greater(types.Uint(8, const=True), types.Uint(8, const=True)),
            types.Uint(8, const=True),
        )
        self.assertEqual(types.greater(types.Bool(), types.Bool()), types.Bool())
        self.assertEqual(types.greater(types.Bool(const=True), types.Bool()), types.Bool())
        self.assertEqual(types.greater(types.Bool(), types.Bool(const=True)), types.Bool())
        self.assertEqual(
            types.greater(types.Bool(const=True), types.Bool(const=True)), types.Bool(const=True)
        )
        self.assertEqual(types.greater(types.Float(), types.Float()), types.Float())
        self.assertEqual(types.greater(types.Float(const=True), types.Float()), types.Float())
        self.assertEqual(types.greater(types.Float(), types.Float(const=True)), types.Float())
        self.assertEqual(
            types.greater(types.Float(const=True), types.Float(const=True)), types.Float(const=True)
        )
        self.assertEqual(types.greater(types.Duration(), types.Duration()), types.Duration())
        self.assertEqual(types.greater(types.Stretch(), types.Duration()), types.Stretch())
        self.assertEqual(types.greater(types.Duration(), types.Stretch()), types.Stretch())
        with self.assertRaisesRegex(TypeError, "no ordering"):
            types.greater(types.Bool(), types.Uint(8))
        with self.assertRaisesRegex(TypeError, "no ordering"):
            types.greater(types.Bool(), types.Float())
        with self.assertRaisesRegex(TypeError, "no ordering"):
            types.greater(types.Bool(), types.Duration())
        with self.assertRaisesRegex(TypeError, "no ordering"):
            types.greater(types.Bool(), types.Stretch())
        with self.assertRaisesRegex(TypeError, "no ordering"):
            types.greater(types.Uint(8), types.Stretch())
        with self.assertRaisesRegex(TypeError, "no ordering"):
            types.greater(types.Uint(8), types.Duration())
        with self.assertRaisesRegex(TypeError, "no ordering"):
            types.greater(types.Uint(16, const=True), types.Uint(8))
        with self.assertRaisesRegex(TypeError, "no ordering"):
            types.greater(types.Uint(8), types.Uint(16, const=True))
        with self.assertRaisesRegex(TypeError, "no ordering"):
            types.greater(types.Float(), types.Duration())
        with self.assertRaisesRegex(TypeError, "no ordering"):
            types.greater(types.Float(), types.Stretch())


class TestTypesCastKind(QiskitTestCase):
    def test_basic_examples(self):
        """This is used extensively throughout the expression construction functions, but since it
        is public API, it should have some direct unit tests as well."""
        # Bool -> Bool
        self.assertIs(types.cast_kind(types.Bool(), types.Bool()), types.CastKind.EQUAL)
        self.assertIs(
            types.cast_kind(types.Bool(const=True), types.Bool(const=True)), types.CastKind.EQUAL
        )
        self.assertIs(
            types.cast_kind(types.Bool(const=True), types.Bool()), types.CastKind.IMPLICIT
        )
        self.assertIs(types.cast_kind(types.Bool(), types.Bool(const=True)), types.CastKind.NONE)

        # Float -> Float
        self.assertIs(types.cast_kind(types.Float(), types.Float()), types.CastKind.EQUAL)
        self.assertIs(
            types.cast_kind(types.Float(const=True), types.Float(const=True)), types.CastKind.EQUAL
        )
        self.assertIs(
            types.cast_kind(types.Float(const=True), types.Float()), types.CastKind.IMPLICIT
        )
        self.assertIs(types.cast_kind(types.Float(), types.Float(const=True)), types.CastKind.NONE)

        # Uint -> Bool
        self.assertIs(types.cast_kind(types.Uint(8), types.Bool()), types.CastKind.IMPLICIT)
        self.assertIs(
            types.cast_kind(types.Uint(8, const=True), types.Bool(const=True)),
            types.CastKind.IMPLICIT,
        )
        self.assertIs(
            types.cast_kind(types.Uint(8, const=True), types.Bool()), types.CastKind.IMPLICIT
        )
        self.assertIs(types.cast_kind(types.Uint(8), types.Bool(const=True)), types.CastKind.NONE)

        # Float -> Bool
        self.assertIs(types.cast_kind(types.Float(), types.Bool()), types.CastKind.DANGEROUS)
        self.assertIs(
            types.cast_kind(types.Float(const=True), types.Bool(const=True)),
            types.CastKind.DANGEROUS,
        )
        self.assertIs(
            types.cast_kind(types.Float(const=True), types.Bool()), types.CastKind.DANGEROUS
        )
        self.assertIs(types.cast_kind(types.Float(), types.Bool(const=True)), types.CastKind.NONE)

        # Bool -> Uint
        self.assertIs(types.cast_kind(types.Bool(), types.Uint(8)), types.CastKind.LOSSLESS)
        self.assertIs(
            types.cast_kind(types.Bool(const=True), types.Uint(8, const=True)),
            types.CastKind.LOSSLESS,
        )
        self.assertIs(
            types.cast_kind(types.Bool(const=True), types.Uint(8)), types.CastKind.LOSSLESS
        )
        self.assertIs(types.cast_kind(types.Bool(), types.Uint(8, const=True)), types.CastKind.NONE)

        # Uint(16) -> Uint(8)
        self.assertIs(types.cast_kind(types.Uint(16), types.Uint(8)), types.CastKind.DANGEROUS)
        self.assertIs(
            types.cast_kind(types.Uint(16, const=True), types.Uint(8, const=True)),
            types.CastKind.DANGEROUS,
        )
        self.assertIs(
            types.cast_kind(types.Uint(16, const=True), types.Uint(8)), types.CastKind.DANGEROUS
        )
        self.assertIs(
            types.cast_kind(types.Uint(16), types.Uint(8, const=True)), types.CastKind.NONE
        )

        # Uint widening
        self.assertIs(types.cast_kind(types.Uint(8), types.Uint(16)), types.CastKind.LOSSLESS)
        self.assertIs(
            types.cast_kind(types.Uint(8, const=True), types.Uint(16, const=True)),
            types.CastKind.LOSSLESS,
        )
        self.assertIs(
            types.cast_kind(types.Uint(8, const=True), types.Uint(16)),
            types.CastKind.LOSSLESS,
        )
        self.assertIs(
            types.cast_kind(types.Uint(8), types.Uint(16, const=True)), types.CastKind.NONE
        )

        # Uint -> Float
        self.assertIs(types.cast_kind(types.Uint(16), types.Float()), types.CastKind.DANGEROUS)
        self.assertIs(
            types.cast_kind(types.Uint(16, const=True), types.Float(const=True)),
            types.CastKind.DANGEROUS,
        )
        self.assertIs(
            types.cast_kind(types.Uint(16, const=True), types.Float()), types.CastKind.DANGEROUS
        )
        self.assertIs(types.cast_kind(types.Uint(16), types.Float(const=True)), types.CastKind.NONE)

        # Float -> Uint(8)
        self.assertIs(types.cast_kind(types.Float(), types.Uint(8)), types.CastKind.DANGEROUS)
        self.assertIs(
            types.cast_kind(types.Float(const=True), types.Uint(8, const=True)),
            types.CastKind.DANGEROUS,
        )
        self.assertIs(
            types.cast_kind(types.Float(const=True), types.Uint(8)),
            types.CastKind.DANGEROUS,
        )
        self.assertIs(
            types.cast_kind(types.Float(), types.Uint(8, const=True)),
            types.CastKind.NONE,
        )

        # Float -> Uint(16)
        self.assertIs(types.cast_kind(types.Float(), types.Uint(16)), types.CastKind.DANGEROUS)
        self.assertIs(
            types.cast_kind(types.Float(const=True), types.Uint(16, const=True)),
            types.CastKind.DANGEROUS,
        )
        self.assertIs(
            types.cast_kind(types.Float(const=True), types.Uint(16)),
            types.CastKind.DANGEROUS,
        )
        self.assertIs(
            types.cast_kind(types.Float(), types.Uint(16, const=True)),
            types.CastKind.NONE,
        )

        # Bool -> Float
        self.assertIs(types.cast_kind(types.Bool(), types.Float()), types.CastKind.LOSSLESS)
        self.assertIs(
            types.cast_kind(types.Bool(const=True), types.Float(const=True)),
            types.CastKind.LOSSLESS,
        )
        self.assertIs(
            types.cast_kind(types.Bool(const=True), types.Float()), types.CastKind.LOSSLESS
        )
        self.assertIs(types.cast_kind(types.Bool(), types.Float(const=True)), types.CastKind.NONE)

        # Uint -> Uint with const qualifiers
        self.assertIs(
            types.cast_kind(types.Uint(16), types.Uint(16, const=True)), types.CastKind.NONE
        )
        self.assertIs(
            types.cast_kind(types.Uint(16, const=True), types.Uint(16)), types.CastKind.IMPLICIT
        )
        self.assertIs(
            types.cast_kind(types.Uint(8), types.Uint(8, const=True)), types.CastKind.NONE
        )
        self.assertIs(
            types.cast_kind(types.Uint(8, const=True), types.Uint(8)), types.CastKind.IMPLICIT
        )

        # Bool -> Uint with const qualifiers
        self.assertIs(types.cast_kind(types.Bool(), types.Uint(8, const=True)), types.CastKind.NONE)
        self.assertIs(
            types.cast_kind(types.Bool(const=True), types.Uint(8)), types.CastKind.LOSSLESS
        )
        self.assertIs(
            types.cast_kind(types.Bool(const=True), types.Uint(8, const=True)),
            types.CastKind.LOSSLESS,
        )

        # Bool -> Float with const qualifiers
        self.assertIs(types.cast_kind(types.Bool(), types.Float(const=True)), types.CastKind.NONE)
        self.assertIs(
            types.cast_kind(types.Bool(const=True), types.Float()), types.CastKind.LOSSLESS
        )
        self.assertIs(
            types.cast_kind(types.Bool(const=True), types.Float(const=True)),
            types.CastKind.LOSSLESS,
        )

        # Duration -> Duration
        self.assertIs(types.cast_kind(types.Duration(), types.Duration()), types.CastKind.EQUAL)

        # Duration -> Stretch (allowed, implicit)
        self.assertIs(types.cast_kind(types.Duration(), types.Stretch()), types.CastKind.IMPLICIT)

        # Stretch -> Stretch
        self.assertIs(types.cast_kind(types.Stretch(), types.Stretch()), types.CastKind.EQUAL)

        # Stretch -> Duration (not allowed)
        self.assertIs(types.cast_kind(types.Stretch(), types.Duration()), types.CastKind.NONE)

        # Duration -> Other types (not allowed, including const variants)
        self.assertIs(types.cast_kind(types.Duration(), types.Bool()), types.CastKind.NONE)
        self.assertIs(
            types.cast_kind(types.Duration(), types.Bool(const=True)), types.CastKind.NONE
        )
        self.assertIs(types.cast_kind(types.Duration(), types.Uint(8)), types.CastKind.NONE)
        self.assertIs(
            types.cast_kind(types.Duration(), types.Uint(8, const=True)), types.CastKind.NONE
        )
        self.assertIs(types.cast_kind(types.Duration(), types.Uint(16)), types.CastKind.NONE)
        self.assertIs(
            types.cast_kind(types.Duration(), types.Uint(16, const=True)), types.CastKind.NONE
        )
        self.assertIs(types.cast_kind(types.Duration(), types.Float()), types.CastKind.NONE)
        self.assertIs(
            types.cast_kind(types.Duration(), types.Float(const=True)), types.CastKind.NONE
        )

        # Stretch -> Other types (not allowed, including const variants)
        self.assertIs(types.cast_kind(types.Stretch(), types.Bool()), types.CastKind.NONE)
        self.assertIs(types.cast_kind(types.Stretch(), types.Bool(const=True)), types.CastKind.NONE)
        self.assertIs(types.cast_kind(types.Stretch(), types.Uint(8)), types.CastKind.NONE)
        self.assertIs(
            types.cast_kind(types.Stretch(), types.Uint(8, const=True)), types.CastKind.NONE
        )
        self.assertIs(types.cast_kind(types.Stretch(), types.Uint(16)), types.CastKind.NONE)
        self.assertIs(
            types.cast_kind(types.Stretch(), types.Uint(16, const=True)), types.CastKind.NONE
        )
        self.assertIs(types.cast_kind(types.Stretch(), types.Float()), types.CastKind.NONE)
        self.assertIs(
            types.cast_kind(types.Stretch(), types.Float(const=True)), types.CastKind.NONE
        )

        # Other types -> Duration (not allowed, including const variants)
        self.assertIs(types.cast_kind(types.Bool(), types.Duration()), types.CastKind.NONE)
        self.assertIs(
            types.cast_kind(types.Bool(const=True), types.Duration()), types.CastKind.NONE
        )
        self.assertIs(types.cast_kind(types.Uint(8), types.Duration()), types.CastKind.NONE)
        self.assertIs(
            types.cast_kind(types.Uint(8, const=True), types.Duration()), types.CastKind.NONE
        )
        self.assertIs(types.cast_kind(types.Uint(16), types.Duration()), types.CastKind.NONE)
        self.assertIs(
            types.cast_kind(types.Uint(16, const=True), types.Duration()), types.CastKind.NONE
        )
        self.assertIs(types.cast_kind(types.Float(), types.Duration()), types.CastKind.NONE)
        self.assertIs(
            types.cast_kind(types.Float(const=True), types.Duration()), types.CastKind.NONE
        )

        # Other types -> Stretch (not allowed, including const variants)
        self.assertIs(types.cast_kind(types.Bool(), types.Stretch()), types.CastKind.NONE)
        self.assertIs(types.cast_kind(types.Bool(const=True), types.Stretch()), types.CastKind.NONE)
        self.assertIs(types.cast_kind(types.Uint(8), types.Stretch()), types.CastKind.NONE)
        self.assertIs(
            types.cast_kind(types.Uint(8, const=True), types.Stretch()), types.CastKind.NONE
        )
        self.assertIs(types.cast_kind(types.Uint(16), types.Stretch()), types.CastKind.NONE)
        self.assertIs(
            types.cast_kind(types.Uint(16, const=True), types.Stretch()), types.CastKind.NONE
        )
        self.assertIs(types.cast_kind(types.Float(), types.Stretch()), types.CastKind.NONE)
        self.assertIs(
            types.cast_kind(types.Float(const=True), types.Stretch()), types.CastKind.NONE
        )
