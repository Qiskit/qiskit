# This code is part of Qiskit.
#
# (C) Copyright IBM 2025
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
import pickle
import unittest

import ddt
import numpy as np

from qiskit.quantum_info import (
    QubitSparsePauli,
    QubitSparsePauliList,
    Pauli,
)

from test import QiskitTestCase  # pylint: disable=wrong-import-order


def single_cases():
    return [
        QubitSparsePauli(""),
        QubitSparsePauli("I" * 10),
        QubitSparsePauli.from_label("IIXIZI"),
        QubitSparsePauli.from_label("ZZYYXX"),
    ]


def single_cases_list():
    return [
        QubitSparsePauliList.empty(0),
        QubitSparsePauliList.empty(10),
        QubitSparsePauliList.from_label("IIXIZI"),
        QubitSparsePauliList.from_list(["YIXZII", "ZZYYXX"]),
        # Includes a duplicate entry.
        QubitSparsePauliList.from_list(["IXZ", "ZZI", "IXZ"]),
    ]


@ddt.ddt
class TestQubitSparsePauli(QiskitTestCase):

    def test_default_constructor_pauli(self):
        data = Pauli("IXYIZ")
        self.assertEqual(QubitSparsePauli(data), QubitSparsePauli.from_pauli(data))
        self.assertEqual(
            QubitSparsePauli(data, num_qubits=data.num_qubits), QubitSparsePauli.from_pauli(data)
        )
        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            QubitSparsePauli(data, num_qubits=data.num_qubits + 1)

        with_phase = Pauli("-jIYYXY")
        self.assertEqual(QubitSparsePauli(with_phase), QubitSparsePauli.from_pauli(with_phase))
        self.assertEqual(
            QubitSparsePauli(with_phase, num_qubits=data.num_qubits),
            QubitSparsePauli.from_pauli(with_phase),
        )

        self.assertEqual(QubitSparsePauli(Pauli("")), QubitSparsePauli.from_pauli(Pauli("")))

    def test_default_constructor_label(self):
        data = "IXIIZ"
        self.assertEqual(QubitSparsePauli(data), QubitSparsePauli.from_label(data))
        self.assertEqual(QubitSparsePauli(data, num_qubits=5), QubitSparsePauli.from_label(data))
        with self.assertRaisesRegex(ValueError, "does not match label"):
            QubitSparsePauli(data, num_qubits=4)
        with self.assertRaisesRegex(ValueError, "does not match label"):
            QubitSparsePauli(data, num_qubits=6)

    def test_default_constructor_sparse_label(self):
        data = ("ZX", (0, 3))
        self.assertEqual(
            QubitSparsePauli(data, num_qubits=5),
            QubitSparsePauli.from_sparse_label(data, num_qubits=5),
        )
        self.assertEqual(
            QubitSparsePauli(data, num_qubits=10),
            QubitSparsePauli.from_sparse_label(data, num_qubits=10),
        )
        with self.assertRaisesRegex(ValueError, "'num_qubits' must be provided"):
            QubitSparsePauli(data)
        self.assertEqual(
            QubitSparsePauli(("", []), num_qubits=5),
            QubitSparsePauli.from_sparse_label(("", []), num_qubits=5),
        )

    def test_from_raw_parts(self):
        # Happiest path: exactly typed inputs.
        num_qubits = 100
        terms = np.full((num_qubits,), QubitSparsePauli.Pauli.Z, dtype=np.uint8)
        indices = np.arange(num_qubits, dtype=np.uint32)
        qubit_sparse_pauli_list = QubitSparsePauli.from_raw_parts(
            num_qubits,
            terms,
            indices,
        )
        self.assertEqual(qubit_sparse_pauli_list.num_qubits, num_qubits)
        np.testing.assert_equal(qubit_sparse_pauli_list.paulis, terms)
        np.testing.assert_equal(qubit_sparse_pauli_list.indices, indices)

        self.assertEqual(
            qubit_sparse_pauli_list,
            QubitSparsePauli.from_raw_parts(
                num_qubits,
                terms,
                indices,
            ),
        )

        # Conversion from array-likes, including mis-typed but compatible arrays.
        qubit_sparse_pauli_list = QubitSparsePauli.from_raw_parts(
            num_qubits,
            tuple(terms),
            qubit_sparse_pauli_list.indices,
        )
        self.assertEqual(qubit_sparse_pauli_list.num_qubits, num_qubits)
        np.testing.assert_equal(qubit_sparse_pauli_list.paulis, terms)
        np.testing.assert_equal(qubit_sparse_pauli_list.indices, indices)

    def test_from_raw_parts_checks_coherence(self):
        with self.assertRaisesRegex(ValueError, "not a valid letter"):
            QubitSparsePauli.from_raw_parts(2, [ord("$")], [0])
        with self.assertRaisesRegex(ValueError, r"`paulis` \(1\) and `indices` \(0\)"):
            QubitSparsePauli.from_raw_parts(2, [QubitSparsePauli.Pauli.Z], [])
        with self.assertRaisesRegex(ValueError, r"`paulis` \(0\) and `indices` \(1\)"):
            QubitSparsePauli.from_raw_parts(2, [], [1])
        with self.assertRaisesRegex(ValueError, r"all qubit indices must be less than the number"):
            QubitSparsePauli.from_raw_parts(4, [1, 2], [0, 4])
        with self.assertRaisesRegex(ValueError, r"all qubit indices must be less than the number"):
            QubitSparsePauli.from_raw_parts(4, [1, 2], [0, 4])

    def test_from_label(self):
        # The label is interpreted like a bitstring, with the right-most item associated with qubit
        # 0, and increasing as we move to the left (like `Pauli`, and other bitstring conventions).
        self.assertEqual(
            # Ruler for counting terms:  dcba9876543210
            QubitSparsePauli.from_label("IXXIIZZIYYIXYZ"),
            QubitSparsePauli.from_raw_parts(
                14,
                [
                    QubitSparsePauli.Pauli.Z,
                    QubitSparsePauli.Pauli.Y,
                    QubitSparsePauli.Pauli.X,
                    QubitSparsePauli.Pauli.Y,
                    QubitSparsePauli.Pauli.Y,
                    QubitSparsePauli.Pauli.Z,
                    QubitSparsePauli.Pauli.Z,
                    QubitSparsePauli.Pauli.X,
                    QubitSparsePauli.Pauli.X,
                ],
                [0, 1, 2, 4, 5, 7, 8, 11, 12],
            ),
        )

    def test_from_label_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            QubitSparsePauli.from_label("I+-$%I")
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Unicode shenangigans.
            QubitSparsePauli.from_label("🐍")

    def test_from_sparse_label(self):
        self.assertEqual(
            QubitSparsePauli.from_sparse_label(("XY", (0, 1)), num_qubits=5),
            QubitSparsePauli.from_label("IIIYX"),
        )
        self.assertEqual(
            QubitSparsePauli.from_sparse_label(("XX", (1, 3)), num_qubits=5),
            QubitSparsePauli.from_label("IXIXI"),
        )
        self.assertEqual(
            QubitSparsePauli.from_sparse_label(("YYZ", (0, 2, 4)), num_qubits=5),
            QubitSparsePauli.from_label("ZIYIY"),
        )

        # The indices should be allowed to be given in unsorted order, but they should be term-wise
        # sorted in the output.
        from_unsorted = QubitSparsePauli.from_sparse_label(("XYZ", (2, 0, 1)), num_qubits=3)
        self.assertEqual(from_unsorted, QubitSparsePauli.from_label("XZY"))
        np.testing.assert_equal(from_unsorted.indices, np.array([0, 1, 2], dtype=np.uint32))

        # Explicit identities should still work, just be skipped over.
        explicit_identity = QubitSparsePauli.from_sparse_label(("ZXI", (0, 1, 2)), num_qubits=10)
        self.assertEqual(
            explicit_identity,
            QubitSparsePauli.from_sparse_label(("XZ", (1, 0)), num_qubits=10),
        )
        np.testing.assert_equal(explicit_identity.indices, np.array([0, 1], dtype=np.uint32))

        explicit_identity = QubitSparsePauli.from_sparse_label(
            ("XYIII", (0, 1, 2, 3, 8)), num_qubits=10
        )
        self.assertEqual(
            explicit_identity,
            QubitSparsePauli.from_sparse_label(("YX", (1, 0)), num_qubits=10),
        )
        np.testing.assert_equal(explicit_identity.indices, np.array([0, 1], dtype=np.uint32))

    def test_from_sparse_list_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            QubitSparsePauli.from_sparse_label(("+$", (2, 1)), num_qubits=8)
        # Unicode shenangigans.  These two should fail with a `ValueError`, but the exact message
        # isn't important.  "\xff" is "ÿ", which is two bytes in UTF-8 (so has a length of 2 in
        # Rust), but has a length of 1 in Python, so try with both a length-1 and length-2 index
        # sequence, and both should still raise `ValueError`.
        with self.assertRaises(ValueError):
            QubitSparsePauli.from_sparse_label(("\xff", (1,)), num_qubits=5)
        with self.assertRaises(ValueError):
            QubitSparsePauli.from_sparse_label(("\xff", (1, 2)), num_qubits=5)

        with self.assertRaisesRegex(
            ValueError, "label with length 2 does not match indices of length 1"
        ):
            QubitSparsePauli.from_sparse_label(("XZ", (0,)), num_qubits=5)
        with self.assertRaisesRegex(
            ValueError, "label with length 2 does not match indices of length 3"
        ):
            QubitSparsePauli.from_sparse_label(("XZ", (0, 1, 2)), num_qubits=5)

        with self.assertRaisesRegex(ValueError, "index 3 is out of range for a 3-qubit operator"):
            QubitSparsePauli.from_sparse_label(("XZY", (0, 1, 3)), num_qubits=3)
        with self.assertRaisesRegex(ValueError, "index 4 is out of range for a 3-qubit operator"):
            QubitSparsePauli.from_sparse_label(("XZY", (0, 1, 4)), num_qubits=3)
        with self.assertRaisesRegex(ValueError, "index 3 is out of range for a 3-qubit operator"):
            # ... even if it's for an explicit identity.
            QubitSparsePauli.from_sparse_label(("XXI", (0, 1, 3)), num_qubits=3)

        with self.assertRaisesRegex(ValueError, "index 3 is duplicated"):
            QubitSparsePauli.from_sparse_label(("XZ", (3, 3)), num_qubits=5)
        with self.assertRaisesRegex(ValueError, "index 3 is duplicated"):
            QubitSparsePauli.from_sparse_label(("XYZXZ", (3, 0, 1, 2, 3)), num_qubits=5)

    def test_from_pauli(self):
        # This function should be infallible provided `Pauli` doesn't change its interface and the
        # user doesn't violate the typing.

        # Simple check that the labels are interpreted in the same order.
        self.assertEqual(
            QubitSparsePauli.from_pauli(Pauli("IIXZI")), QubitSparsePauli.from_label("IIXZI")
        )

        # `Pauli` accepts a phase in its label, which gets dropped
        self.assertEqual(
            QubitSparsePauli.from_pauli(Pauli("iIXZIX")),
            QubitSparsePauli.from_label("IXZIX"),
        )
        self.assertEqual(
            QubitSparsePauli.from_pauli(Pauli("-iIXZIX")),
            QubitSparsePauli.from_label("IXZIX"),
        )
        self.assertEqual(
            QubitSparsePauli.from_pauli(Pauli("-IXZIX")),
            QubitSparsePauli.from_label("IXZIX"),
        )

        # `Pauli` has its internal phase convention for how it stores `Y`; we should get this right
        # regardless of how many Ys are in the label, or if there's a phase.
        paulis = {"IXYZ" * n: Pauli("IXYZ" * n) for n in range(1, 5)}
        from_paulis, from_labels = zip(
            *(
                (QubitSparsePauli.from_pauli(pauli), QubitSparsePauli.from_label(label))
                for label, pauli in paulis.items()
            )
        )
        self.assertEqual(from_paulis, from_labels)

        phased_paulis = {"IXYZ" * n: Pauli("j" + "IXYZ" * n) for n in range(1, 5)}
        from_paulis, from_lists = zip(
            *(
                (QubitSparsePauli.from_pauli(pauli), QubitSparsePauli.from_label(label))
                for label, pauli in phased_paulis.items()
            )
        )
        self.assertEqual(from_paulis, from_lists)

    def test_default_constructor_failed_inference(self):
        with self.assertRaises(TypeError):
            QubitSparsePauli(5, num_qubits=5)

    def test_num_qubits(self):
        self.assertEqual(QubitSparsePauli("").num_qubits, 0)
        self.assertEqual(QubitSparsePauli("I" * 10).num_qubits, 10)

    def test_pauli_enum(self):
        # These are very explicit tests that effectively just duplicate magic numbers, but the point
        # is that those magic numbers are required to be constant as their values are part of the
        # public interface.

        self.assertEqual(
            set(QubitSparsePauli.Pauli),
            {
                QubitSparsePauli.Pauli.X,
                QubitSparsePauli.Pauli.Y,
                QubitSparsePauli.Pauli.Z,
            },
        )
        # All the enumeration items should also be integers.
        self.assertIsInstance(QubitSparsePauli.Pauli.X, int)
        values = {
            "X": 0b10,
            "Y": 0b11,
            "Z": 0b01,
        }
        self.assertEqual({name: getattr(QubitSparsePauli.Pauli, name) for name in values}, values)

        # The single-character label aliases can be accessed with index notation.
        labels = {
            "X": QubitSparsePauli.Pauli.X,
            "Y": QubitSparsePauli.Pauli.Y,
            "Z": QubitSparsePauli.Pauli.Z,
        }
        self.assertEqual({label: QubitSparsePauli.Pauli[label] for label in labels}, labels)
        # The `label` property returns known values.
        self.assertEqual({pauli.label: pauli for pauli in QubitSparsePauli.Pauli}, labels)

    @ddt.idata(single_cases())
    def test_pickle(self, qubit_sparse_pauli):
        self.assertEqual(qubit_sparse_pauli, copy.copy(qubit_sparse_pauli))
        self.assertIsNot(qubit_sparse_pauli, copy.copy(qubit_sparse_pauli))
        self.assertEqual(qubit_sparse_pauli, copy.deepcopy(qubit_sparse_pauli))
        self.assertEqual(qubit_sparse_pauli, pickle.loads(pickle.dumps(qubit_sparse_pauli)))

    @ddt.data(
        # This is every combination of (0, 1, many) for (terms, qubits, non-identites per term).
        QubitSparsePauli.from_label("IIXIZI"),
        QubitSparsePauli.from_label("X"),
    )
    def test_repr(self, data):
        # The purpose of this is just to test that the `repr` doesn't crash, rather than asserting
        # that it has any particular form.
        self.assertIsInstance(repr(data), str)
        self.assertIn("QubitSparsePauli", repr(data))

    @ddt.idata(single_cases())
    def test_copy(self, qubit_sparse_pauli):
        self.assertEqual(qubit_sparse_pauli, qubit_sparse_pauli.copy())
        self.assertIsNot(qubit_sparse_pauli, qubit_sparse_pauli.copy())

    def test_equality(self):
        sparse_data = ("XYY", (3, 1, 0))
        pauli = QubitSparsePauli.from_sparse_label(sparse_data, num_qubits=5)
        self.assertEqual(pauli, pauli.copy())
        # Take care that Rust space allows multiple views onto the same object.
        self.assertEqual(pauli, pauli)

        # Comparison to some other object shouldn't fail.
        self.assertNotEqual(pauli, None)

        # Difference in qubit count.
        self.assertNotEqual(
            pauli, QubitSparsePauli.from_sparse_label(sparse_data, num_qubits=pauli.num_qubits + 1)
        )

        # Difference in bit terms.
        self.assertNotEqual(
            QubitSparsePauli.from_label("IIXZI"),
            QubitSparsePauli.from_label("IIYZI"),
        )
        self.assertNotEqual(
            QubitSparsePauli.from_label("XXYYZ"),
            QubitSparsePauli.from_label("XXYYY"),
        )

        # Difference in indices.
        self.assertNotEqual(
            QubitSparsePauli.from_label("IIXZI"),
            QubitSparsePauli.from_label("IXIZI"),
        )
        self.assertNotEqual(
            QubitSparsePauli.from_label("XIYYZ"),
            QubitSparsePauli.from_label("IXYYZ"),
        )

    def test_attributes_sequence(self):
        """Test attributes of the `Sequence` protocol."""
        # Length
        pauli = QubitSparsePauli.from_label("XZY")
        self.assertEqual(len(pauli.indices), 3)
        self.assertEqual(len(pauli.paulis), 3)

        # Iteration
        self.assertEqual(tuple(pauli.indices), (0, 1, 2))
        # multiple iteration through same object
        paulis = pauli.paulis
        self.assertEqual(set(paulis), {QubitSparsePauli.Pauli[x] for x in "XZY"})

        # Implicit iteration methods.
        self.assertIn(QubitSparsePauli.Pauli.Y, pauli.paulis)
        self.assertNotIn(4, pauli.indices)

        # Index by scalar
        self.assertEqual(pauli.indices[-1], 2)
        self.assertEqual(pauli.paulis[0], QubitSparsePauli.Pauli.Y)

        # Index by slice.  This is API guaranteed to be a Numpy array to make it easier to
        # manipulate subslices with mathematic operations.
        self.assertIsInstance(pauli.indices[::-1], np.ndarray)
        np.testing.assert_array_equal(
            pauli.indices[::-1],
            np.array([2, 1, 0], dtype=np.uint32),
            strict=True,
        )
        self.assertIsInstance(pauli.paulis[0:2], np.ndarray)
        np.testing.assert_array_equal(
            pauli.paulis[0:2],
            np.array([QubitSparsePauli.Pauli.Y, QubitSparsePauli.Pauli.Z], dtype=np.uint8),
            strict=True,
        )

    def test_attributes_to_array(self):
        pauli = QubitSparsePauli.from_label("XZY")

        # Natural dtypes.
        np.testing.assert_array_equal(
            pauli.indices, np.array([0, 1, 2], dtype=np.uint32), strict=True
        )
        np.testing.assert_array_equal(
            pauli.paulis,
            np.array([QubitSparsePauli.Pauli[x] for x in "YZX"], dtype=np.uint8),
            strict=True,
        )

        # Cast dtypes.
        np.testing.assert_array_equal(
            np.array(pauli.indices, dtype=np.uint8),
            np.array([0, 1, 2], dtype=np.uint8),
            strict=True,
        )


@ddt.ddt
class TestQubitSparsePauliList(QiskitTestCase):
    def test_default_constructor_pauli(self):
        data = Pauli("IXYIZ")
        self.assertEqual(QubitSparsePauliList(data), QubitSparsePauliList.from_pauli(data))
        self.assertEqual(
            QubitSparsePauliList(data, num_qubits=data.num_qubits),
            QubitSparsePauliList.from_pauli(data),
        )
        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            QubitSparsePauliList(data, num_qubits=data.num_qubits + 1)

        with_phase = Pauli("-jIYYXY")
        self.assertEqual(
            QubitSparsePauliList(with_phase), QubitSparsePauliList.from_pauli(with_phase)
        )
        self.assertEqual(
            QubitSparsePauliList(with_phase, num_qubits=data.num_qubits),
            QubitSparsePauliList.from_pauli(with_phase),
        )

        self.assertEqual(
            QubitSparsePauliList(Pauli("")), QubitSparsePauliList.from_pauli(Pauli(""))
        )

    def test_default_constructor_list(self):
        data = ["IXIIZ", "XIXII", "IIXYI"]
        self.assertEqual(QubitSparsePauliList(data), QubitSparsePauliList.from_list(data))
        self.assertEqual(
            QubitSparsePauliList(data, num_qubits=5), QubitSparsePauliList.from_list(data)
        )
        with self.assertRaisesRegex(ValueError, "label with length 5 cannot be added"):
            QubitSparsePauliList(data, num_qubits=4)
        with self.assertRaisesRegex(ValueError, "label with length 5 cannot be added"):
            QubitSparsePauliList(data, num_qubits=6)
        self.assertEqual(
            QubitSparsePauliList([], num_qubits=5), QubitSparsePauliList.from_list([], num_qubits=5)
        )

    def test_default_constructor_sparse_list(self):
        data = [("ZX", (0, 3)), ("XY", (2, 4)), ("ZY", (2, 1))]
        self.assertEqual(
            QubitSparsePauliList(data, num_qubits=5),
            QubitSparsePauliList.from_sparse_list(data, num_qubits=5),
        )
        self.assertEqual(
            QubitSparsePauliList(data, num_qubits=10),
            QubitSparsePauliList.from_sparse_list(data, num_qubits=10),
        )
        with self.assertRaisesRegex(ValueError, "'num_qubits' must be provided"):
            QubitSparsePauliList(data)
        self.assertEqual(
            QubitSparsePauliList([], num_qubits=5),
            QubitSparsePauliList.from_sparse_list([], num_qubits=5),
        )

    def test_default_constructor_label(self):
        data = "IIXIXXIZZYYIYZ"
        self.assertEqual(QubitSparsePauliList(data), QubitSparsePauliList.from_label(data))
        self.assertEqual(
            QubitSparsePauliList(data, num_qubits=len(data)), QubitSparsePauliList.from_label(data)
        )
        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            QubitSparsePauliList(data, num_qubits=len(data) + 1)

    def test_default_constructor_copy(self):
        base = QubitSparsePauliList.from_list(["IXIZIY", "XYZIII"])
        copied = QubitSparsePauliList(base)
        self.assertEqual(base, copied)
        self.assertIsNot(base, copied)

        # Modifications to `copied` don't propagate back.
        copied.indices[0] = 1
        self.assertNotEqual(base, copied)

        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            QubitSparsePauliList(base, num_qubits=base.num_qubits + 1)

    def test_default_constructor_term(self):
        expected = QubitSparsePauliList.from_list(["IIZXII"])
        self.assertEqual(QubitSparsePauliList(expected[0]), expected)

    def test_default_constructor_term_iterable(self):
        expected = QubitSparsePauliList.from_list(["IIZXII", "IIIIII"])
        terms = [expected[0], expected[1]]
        self.assertEqual(QubitSparsePauliList(list(terms)), expected)
        self.assertEqual(QubitSparsePauliList(tuple(terms)), expected)
        self.assertEqual(QubitSparsePauliList(term for term in terms), expected)

    def test_from_raw_parts(self):
        # Happiest path: exactly typed inputs.
        num_qubits = 100
        terms = np.full((num_qubits,), QubitSparsePauliList.Pauli.Z, dtype=np.uint8)
        indices = np.arange(num_qubits, dtype=np.uint32)
        boundaries = np.arange(num_qubits + 1, dtype=np.uintp)
        qubit_sparse_pauli_list = QubitSparsePauliList.from_raw_parts(
            num_qubits, terms, indices, boundaries
        )
        self.assertEqual(qubit_sparse_pauli_list.num_qubits, num_qubits)
        np.testing.assert_equal(qubit_sparse_pauli_list.paulis, terms)
        np.testing.assert_equal(qubit_sparse_pauli_list.indices, indices)
        np.testing.assert_equal(qubit_sparse_pauli_list.boundaries, boundaries)

        self.assertEqual(
            qubit_sparse_pauli_list,
            QubitSparsePauliList.from_raw_parts(
                num_qubits, terms, indices, boundaries, check=False
            ),
        )

        # Conversion from array-likes, including mis-typed but compatible arrays.
        qubit_sparse_pauli_list = QubitSparsePauliList.from_raw_parts(
            num_qubits,
            tuple(terms),
            qubit_sparse_pauli_list.indices,
            boundaries.astype(np.int16),
        )
        self.assertEqual(qubit_sparse_pauli_list.num_qubits, num_qubits)
        np.testing.assert_equal(qubit_sparse_pauli_list.paulis, terms)
        np.testing.assert_equal(qubit_sparse_pauli_list.indices, indices)
        np.testing.assert_equal(qubit_sparse_pauli_list.boundaries, boundaries)

        # Construction of an empty list.
        self.assertEqual(
            QubitSparsePauliList.from_raw_parts(10, [], [], [0]), QubitSparsePauliList.empty(10)
        )

        # Construction of an operator with an intermediate identity term.  For the initial
        # constructor tests, it's hard to check anything more than the construction succeeded.
        self.assertEqual(
            QubitSparsePauliList.from_raw_parts(10, [1, 3, 2], [0, 1, 2], [0, 1, 1, 3]).num_terms,
            3,
        )

    def test_from_raw_parts_checks_coherence(self):
        with self.assertRaisesRegex(ValueError, "not a valid letter"):
            QubitSparsePauliList.from_raw_parts(2, [ord("$")], [0], [0, 1])
        with self.assertRaisesRegex(ValueError, r"`paulis` \(1\) and `indices` \(0\)"):
            QubitSparsePauliList.from_raw_parts(2, [QubitSparsePauliList.Pauli.Z], [], [0, 1])
        with self.assertRaisesRegex(ValueError, r"`paulis` \(0\) and `indices` \(1\)"):
            QubitSparsePauliList.from_raw_parts(2, [], [1], [0, 1])
        with self.assertRaisesRegex(ValueError, r"the first item of `boundaries` \(1\) must be 0"):
            QubitSparsePauliList.from_raw_parts(2, [QubitSparsePauliList.Pauli.Z], [0], [1, 1])
        with self.assertRaisesRegex(ValueError, r"the last item of `boundaries` \(2\)"):
            QubitSparsePauliList.from_raw_parts(2, [1], [0], [0, 2])
        with self.assertRaisesRegex(ValueError, r"the last item of `boundaries` \(1\)"):
            QubitSparsePauliList.from_raw_parts(2, [1, 2], [0, 1], [0, 1])
        with self.assertRaisesRegex(ValueError, r"the last item of `boundaries` \(0\)"):
            QubitSparsePauliList.from_raw_parts(2, [QubitSparsePauliList.Pauli.Z], [0], [0])
        with self.assertRaisesRegex(ValueError, r"all qubit indices must be less than the number"):
            QubitSparsePauliList.from_raw_parts(4, [1, 2], [0, 4], [0, 2])
        with self.assertRaisesRegex(ValueError, r"all qubit indices must be less than the number"):
            QubitSparsePauliList.from_raw_parts(4, [1, 2], [0, 4], [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "the values in `boundaries` include backwards"):
            QubitSparsePauliList.from_raw_parts(5, [1, 2, 3, 2], [0, 1, 2, 3], [0, 2, 1, 4])
        with self.assertRaisesRegex(
            ValueError, "the values in `indices` are not term-wise increasing"
        ):
            QubitSparsePauliList.from_raw_parts(4, [1, 2], [1, 0], [0, 2])

        # There's no test of attempting to pass incoherent data and `check=False` because that
        # permits undefined behaviour in Rust (it's unsafe), so all bets would be off.

    def test_from_label(self):
        # The label is interpreted like a bitstring, with the right-most item associated with qubit
        # 0, and increasing as we move to the left (like `Pauli`, and other bitstring conventions).
        self.assertEqual(
            # Ruler for counting terms:  dcba9876543210
            QubitSparsePauliList.from_label("IXXIIZZIYYIXYZ"),
            QubitSparsePauliList.from_raw_parts(
                14,
                [
                    QubitSparsePauliList.Pauli.Z,
                    QubitSparsePauliList.Pauli.Y,
                    QubitSparsePauliList.Pauli.X,
                    QubitSparsePauliList.Pauli.Y,
                    QubitSparsePauliList.Pauli.Y,
                    QubitSparsePauliList.Pauli.Z,
                    QubitSparsePauliList.Pauli.Z,
                    QubitSparsePauliList.Pauli.X,
                    QubitSparsePauliList.Pauli.X,
                ],
                [0, 1, 2, 4, 5, 7, 8, 11, 12],
                [0, 9],
            ),
        )

    def test_from_label_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            QubitSparsePauliList.from_label("I+-$%I")
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Unicode shenangigans.
            QubitSparsePauliList.from_label("🐍")

    def test_from_list(self):
        label = "IXYIZZY"
        self.assertEqual(
            QubitSparsePauliList.from_list([label]), QubitSparsePauliList.from_label(label)
        )
        self.assertEqual(
            QubitSparsePauliList.from_list([label], num_qubits=len(label)),
            QubitSparsePauliList.from_label(label),
        )
        self.assertEqual(
            QubitSparsePauliList.from_list([label]),
            QubitSparsePauliList.from_raw_parts(
                len(label),
                [
                    QubitSparsePauliList.Pauli.Y,
                    QubitSparsePauliList.Pauli.Z,
                    QubitSparsePauliList.Pauli.Z,
                    QubitSparsePauliList.Pauli.Y,
                    QubitSparsePauliList.Pauli.X,
                ],
                [0, 1, 2, 4, 5],
                [0, 5],
            ),
        )
        self.assertEqual(
            QubitSparsePauliList.from_list([label], num_qubits=len(label)),
            QubitSparsePauliList.from_raw_parts(
                len(label),
                [
                    QubitSparsePauliList.Pauli.Y,
                    QubitSparsePauliList.Pauli.Z,
                    QubitSparsePauliList.Pauli.Z,
                    QubitSparsePauliList.Pauli.Y,
                    QubitSparsePauliList.Pauli.X,
                ],
                [0, 1, 2, 4, 5],
                [0, 5],
            ),
        )

        self.assertEqual(
            QubitSparsePauliList.from_list(["IIIXZI", "XXIIII"]),
            QubitSparsePauliList.from_raw_parts(
                6,
                [
                    QubitSparsePauliList.Pauli.Z,
                    QubitSparsePauliList.Pauli.X,
                    QubitSparsePauliList.Pauli.X,
                    QubitSparsePauliList.Pauli.X,
                ],
                [1, 2, 4, 5],
                [0, 2, 4],
            ),
        )

        self.assertEqual(
            QubitSparsePauliList.from_list([], num_qubits=5), QubitSparsePauliList.empty(5)
        )
        self.assertEqual(
            QubitSparsePauliList.from_list([], num_qubits=0), QubitSparsePauliList.empty(0)
        )

    def test_from_list_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            QubitSparsePauliList.from_list(["XZIIZY", "I+-$%I"])
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Unicode shenangigans.
            QubitSparsePauliList.from_list(["🐍"])
        with self.assertRaisesRegex(ValueError, "label with length 4 cannot be added"):
            QubitSparsePauliList.from_list(["IIZ", "IIXI"])
        with self.assertRaisesRegex(ValueError, "label with length 2 cannot be added"):
            QubitSparsePauliList.from_list(["IIZ", "II"])
        with self.assertRaisesRegex(ValueError, "label with length 3 cannot be added"):
            QubitSparsePauliList.from_list(["IIZ", "IXI"], num_qubits=2)
        with self.assertRaisesRegex(ValueError, "label with length 3 cannot be added"):
            QubitSparsePauliList.from_list(["IIZ", "IXI"], num_qubits=4)
        with self.assertRaisesRegex(ValueError, "cannot construct.*without knowing `num_qubits`"):
            QubitSparsePauliList.from_list([])

    def test_from_sparse_list(self):
        self.assertEqual(
            QubitSparsePauliList.from_sparse_list(
                [
                    ("XY", (0, 1)),
                    ("XX", (1, 3)),
                    ("YYZ", (0, 2, 4)),
                ],
                num_qubits=5,
            ),
            QubitSparsePauliList.from_list(["IIIYX", "IXIXI", "ZIYIY"]),
        )

        # The indices should be allowed to be given in unsorted order, but they should be term-wise
        # sorted in the output.
        from_unsorted = QubitSparsePauliList.from_sparse_list(
            [
                ("XYZ", (2, 1, 0)),
                ("XYZ", (2, 0, 1)),
            ],
            num_qubits=3,
        )
        self.assertEqual(from_unsorted, QubitSparsePauliList.from_list(["XYZ", "XZY"]))
        np.testing.assert_equal(
            from_unsorted.indices, np.array([0, 1, 2, 0, 1, 2], dtype=np.uint32)
        )

        # Explicit identities should still work, just be skipped over.
        explicit_identity = QubitSparsePauliList.from_sparse_list(
            [
                ("ZXI", (0, 1, 2)),
                ("XYIII", (0, 1, 2, 3, 8)),
            ],
            num_qubits=10,
        )
        self.assertEqual(
            explicit_identity,
            QubitSparsePauliList.from_sparse_list([("XZ", (1, 0)), ("YX", (1, 0))], num_qubits=10),
        )
        np.testing.assert_equal(explicit_identity.indices, np.array([0, 1, 0, 1], dtype=np.uint32))

        self.assertEqual(
            QubitSparsePauliList.from_sparse_list([], num_qubits=1_000_000),
            QubitSparsePauliList.empty(1_000_000),
        )
        self.assertEqual(
            QubitSparsePauliList.from_sparse_list([], num_qubits=0),
            QubitSparsePauliList.empty(0),
        )

    def test_from_sparse_list_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            QubitSparsePauliList.from_sparse_list(
                [("XZZY", (5, 3, 1, 0)), ("+$", (2, 1))], num_qubits=8
            )
        # Unicode shenangigans.  These two should fail with a `ValueError`, but the exact message
        # isn't important.  "\xff" is "ÿ", which is two bytes in UTF-8 (so has a length of 2 in
        # Rust), but has a length of 1 in Python, so try with both a length-1 and length-2 index
        # sequence, and both should still raise `ValueError`.
        with self.assertRaises(ValueError):
            QubitSparsePauliList.from_sparse_list([("\xff", (1,))], num_qubits=5)
        with self.assertRaises(ValueError):
            QubitSparsePauliList.from_sparse_list([("\xff", (1, 2))], num_qubits=5)

        with self.assertRaisesRegex(ValueError, "label with length 2 does not match indices"):
            QubitSparsePauliList.from_sparse_list([("XZ", (0,))], num_qubits=5)
        with self.assertRaisesRegex(ValueError, "label with length 2 does not match indices"):
            QubitSparsePauliList.from_sparse_list([("XZ", (0, 1, 2))], num_qubits=5)

        with self.assertRaisesRegex(ValueError, "index 3 is out of range for a 3-qubit operator"):
            QubitSparsePauliList.from_sparse_list([("XZY", (0, 1, 3))], num_qubits=3)
        with self.assertRaisesRegex(ValueError, "index 4 is out of range for a 3-qubit operator"):
            QubitSparsePauliList.from_sparse_list([("XZY", (0, 1, 4))], num_qubits=3)
        with self.assertRaisesRegex(ValueError, "index 3 is out of range for a 3-qubit operator"):
            # ... even if it's for an explicit identity.
            QubitSparsePauliList.from_sparse_list([("XXI", (0, 1, 3))], num_qubits=3)

        with self.assertRaisesRegex(ValueError, "index 3 is duplicated"):
            QubitSparsePauliList.from_sparse_list([("XZ", (3, 3))], num_qubits=5)
        with self.assertRaisesRegex(ValueError, "index 3 is duplicated"):
            QubitSparsePauliList.from_sparse_list([("XYZXZ", (3, 0, 1, 2, 3))], num_qubits=5)

    def test_from_pauli(self):
        # This function should be infallible provided `Pauli` doesn't change its interface and the
        # user doesn't violate the typing.

        # Simple check that the labels are interpreted in the same order.
        self.assertEqual(
            QubitSparsePauliList.from_pauli(Pauli("IIXZI")),
            QubitSparsePauliList.from_label("IIXZI"),
        )

        # `Pauli` accepts a phase in its label, which gets dropped
        self.assertEqual(
            QubitSparsePauliList.from_pauli(Pauli("iIXZIX")),
            QubitSparsePauliList.from_list(["IXZIX"]),
        )
        self.assertEqual(
            QubitSparsePauliList.from_pauli(Pauli("-iIXZIX")),
            QubitSparsePauliList.from_list(["IXZIX"]),
        )
        self.assertEqual(
            QubitSparsePauliList.from_pauli(Pauli("-IXZIX")),
            QubitSparsePauliList.from_list(["IXZIX"]),
        )

        # `Pauli` has its internal phase convention for how it stores `Y`; we should get this right
        # regardless of how many Ys are in the label, or if there's a phase.
        paulis = {"IXYZ" * n: Pauli("IXYZ" * n) for n in range(1, 5)}
        from_paulis, from_labels = zip(
            *(
                (QubitSparsePauliList.from_pauli(pauli), QubitSparsePauliList.from_label(label))
                for label, pauli in paulis.items()
            )
        )
        self.assertEqual(from_paulis, from_labels)

        phased_paulis = {"IXYZ" * n: Pauli("j" + "IXYZ" * n) for n in range(1, 5)}
        from_paulis, from_lists = zip(
            *(
                (QubitSparsePauliList.from_pauli(pauli), QubitSparsePauliList.from_list([label]))
                for label, pauli in phased_paulis.items()
            )
        )
        self.assertEqual(from_paulis, from_lists)

    def test_from_qubit_sparse_paulis(self):
        self.assertEqual(
            QubitSparsePauliList.from_qubit_sparse_paulis([], num_qubits=5),
            QubitSparsePauliList.empty(5),
        )
        self.assertEqual(
            QubitSparsePauliList.from_qubit_sparse_paulis((), num_qubits=0),
            QubitSparsePauliList.empty(0),
        )
        self.assertEqual(
            QubitSparsePauliList.from_qubit_sparse_paulis((None for _ in []), num_qubits=3),
            QubitSparsePauliList.empty(3),
        )

        expected = QubitSparsePauliList.from_sparse_list(
            [
                ("XYZ", (4, 2, 1)),
                ("XXYY", (8, 5, 3, 2)),
                ("ZZ", (5, 0)),
            ],
            num_qubits=10,
        )
        self.assertEqual(QubitSparsePauliList.from_qubit_sparse_paulis(list(expected)), expected)
        self.assertEqual(QubitSparsePauliList.from_qubit_sparse_paulis(tuple(expected)), expected)
        self.assertEqual(
            QubitSparsePauliList.from_qubit_sparse_paulis(term for term in expected), expected
        )
        self.assertEqual(
            QubitSparsePauliList.from_qubit_sparse_paulis(
                (term for term in expected), num_qubits=expected.num_qubits
            ),
            expected,
        )

    def test_from_qubit_sparse_paulis_failures(self):
        with self.assertRaisesRegex(ValueError, "cannot construct.*without knowing `num_qubits`"):
            QubitSparsePauliList.from_qubit_sparse_paulis([])

        left, right = (
            QubitSparsePauliList(["IIXYI"])[0],
            QubitSparsePauliList(["IIIIIIIIX"])[0],
        )
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits"):
            QubitSparsePauliList.from_qubit_sparse_paulis([left, right])
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits"):
            QubitSparsePauliList.from_qubit_sparse_paulis([left], num_qubits=100)

    def test_default_constructor_failed_inference(self):
        with self.assertRaises(TypeError):
            # Mixed dense/sparse list.
            QubitSparsePauliList(["IIXIZ", ("IZ", (2, 3))], num_qubits=5)

    def test_num_qubits(self):
        self.assertEqual(QubitSparsePauliList.empty(0).num_qubits, 0)
        self.assertEqual(QubitSparsePauliList.empty(10).num_qubits, 10)

    def test_num_terms(self):
        self.assertEqual(QubitSparsePauliList.empty(0).num_terms, 0)
        self.assertEqual(QubitSparsePauliList.empty(10).num_terms, 0)
        self.assertEqual(QubitSparsePauliList.from_list(["IIIXIZ", "YYXXII"]).num_terms, 2)

    def test_empty(self):
        empty_5 = QubitSparsePauliList.empty(5)
        self.assertEqual(empty_5.num_qubits, 5)
        np.testing.assert_equal(empty_5.paulis, np.array([], dtype=np.uint8))
        np.testing.assert_equal(empty_5.indices, np.array([], dtype=np.uint32))
        np.testing.assert_equal(empty_5.boundaries, np.array([0], dtype=np.uintp))

        empty_0 = QubitSparsePauliList.empty(0)
        self.assertEqual(empty_0.num_qubits, 0)
        np.testing.assert_equal(empty_0.paulis, np.array([], dtype=np.uint8))
        np.testing.assert_equal(empty_0.indices, np.array([], dtype=np.uint32))
        np.testing.assert_equal(empty_0.boundaries, np.array([0], dtype=np.uintp))

    def test_len(self):
        self.assertEqual(len(QubitSparsePauliList.empty(0)), 0)
        self.assertEqual(len(QubitSparsePauliList.empty(10)), 0)
        self.assertEqual(len(QubitSparsePauliList.from_list(["IIIXIZ", "YYXXII"])), 2)

    def test_pauli_enum(self):
        # These are very explicit tests that effectively just duplicate magic numbers, but the point
        # is that those magic numbers are required to be constant as their values are part of the
        # public interface.

        self.assertEqual(
            set(QubitSparsePauliList.Pauli),
            {
                QubitSparsePauliList.Pauli.X,
                QubitSparsePauliList.Pauli.Y,
                QubitSparsePauliList.Pauli.Z,
            },
        )
        # All the enumeration items should also be integers.
        self.assertIsInstance(QubitSparsePauliList.Pauli.X, int)
        values = {
            "X": 0b10,
            "Y": 0b11,
            "Z": 0b01,
        }
        self.assertEqual(
            {name: getattr(QubitSparsePauliList.Pauli, name) for name in values}, values
        )

        # The single-character label aliases can be accessed with index notation.
        labels = {
            "X": QubitSparsePauliList.Pauli.X,
            "Y": QubitSparsePauliList.Pauli.Y,
            "Z": QubitSparsePauliList.Pauli.Z,
        }
        self.assertEqual({label: QubitSparsePauliList.Pauli[label] for label in labels}, labels)
        # The `label` property returns known values.
        self.assertEqual({pauli.label: pauli for pauli in QubitSparsePauliList.Pauli}, labels)

    @ddt.idata(single_cases_list())
    def test_pickle(self, qubit_sparse_pauli_list):
        self.assertEqual(qubit_sparse_pauli_list, copy.copy(qubit_sparse_pauli_list))
        self.assertIsNot(qubit_sparse_pauli_list, copy.copy(qubit_sparse_pauli_list))
        self.assertEqual(qubit_sparse_pauli_list, copy.deepcopy(qubit_sparse_pauli_list))
        self.assertEqual(
            qubit_sparse_pauli_list, pickle.loads(pickle.dumps(qubit_sparse_pauli_list))
        )

    @ddt.data(
        # This is every combination of (0, 1, many) for (terms, qubits, non-identites per term).
        QubitSparsePauliList.empty(0),
        QubitSparsePauliList.empty(1),
        QubitSparsePauliList.empty(10),
        QubitSparsePauliList.from_label("IIXIZI"),
        QubitSparsePauliList.from_label("X"),
        QubitSparsePauliList.from_list(["YIXZII"]),
        QubitSparsePauliList.from_list(["YIXZII", "ZZYYXX"]),
    )
    def test_repr(self, data):
        # The purpose of this is just to test that the `repr` doesn't crash, rather than asserting
        # that it has any particular form.
        self.assertIsInstance(repr(data), str)
        self.assertIn("QubitSparsePauliList", repr(data))

    @ddt.idata(single_cases_list())
    def test_copy(self, qubit_sparse_pauli_list):
        self.assertEqual(qubit_sparse_pauli_list, qubit_sparse_pauli_list.copy())
        self.assertIsNot(qubit_sparse_pauli_list, qubit_sparse_pauli_list.copy())

    def test_equality(self):
        sparse_data = [("XZ", (1, 0)), ("XYY", (3, 1, 0))]
        pauli_list = QubitSparsePauliList.from_sparse_list(sparse_data, num_qubits=5)
        self.assertEqual(pauli_list, pauli_list.copy())
        # Take care that Rust space allows multiple views onto the same object.
        self.assertEqual(pauli_list, pauli_list)

        # Comparison to some other object shouldn't fail.
        self.assertNotEqual(pauli_list, None)

        # Difference in qubit count.
        self.assertNotEqual(
            pauli_list,
            QubitSparsePauliList.from_sparse_list(
                sparse_data, num_qubits=pauli_list.num_qubits + 1
            ),
        )
        self.assertNotEqual(QubitSparsePauliList.empty(2), QubitSparsePauliList.empty(3))

        # Difference in bit terms.
        self.assertNotEqual(
            QubitSparsePauliList.from_list(["IIXZI", "XXYYZ"]),
            QubitSparsePauliList.from_list(["IIYZI", "XXYYZ"]),
        )
        self.assertNotEqual(
            QubitSparsePauliList.from_list(["IIXZI", "XXYYZ"]),
            QubitSparsePauliList.from_list(["IIXZI", "XXYYY"]),
        )

        # Difference in indices.
        self.assertNotEqual(
            QubitSparsePauliList.from_list(["IIXZI", "XXYYZ"]),
            QubitSparsePauliList.from_list(["IXIZI", "XXYYZ"]),
        )
        self.assertNotEqual(
            QubitSparsePauliList.from_list(["IIXZI", "XIYYZ"]),
            QubitSparsePauliList.from_list(["IIXZI", "IXYYZ"]),
        )

        # Difference in boundaries.
        self.assertNotEqual(
            QubitSparsePauliList.from_sparse_list([("XZ", (0, 1)), ("XX", (2, 3))], num_qubits=5),
            QubitSparsePauliList.from_sparse_list([("XZX", (0, 1, 2)), ("X", (3,))], num_qubits=5),
        )

    def test_write_into_attributes_scalar(self):

        paulis = QubitSparsePauliList.from_sparse_list(
            [("XZ", (0, 1)), ("XX", (2, 3))], num_qubits=8
        )
        paulis.paulis[0] = QubitSparsePauliList.Pauli.Y
        paulis.paulis[3] = QubitSparsePauliList.Pauli.Z
        self.assertEqual(
            paulis,
            QubitSparsePauliList.from_sparse_list([("YZ", (0, 1)), ("XZ", (2, 3))], num_qubits=8),
        )

        indices = QubitSparsePauliList.from_sparse_list(
            [("XZ", (0, 1)), ("XX", (2, 3))], num_qubits=8
        )
        # These two sets keep the generator in term-wise increasing order.  We don't test what
        # happens if somebody violates the Rust-space requirement to be term-wise increasing.
        indices.indices[1] = 4
        indices.indices[3] = 7
        self.assertEqual(
            indices,
            QubitSparsePauliList.from_sparse_list([("XZ", (0, 4)), ("XX", (2, 7))], num_qubits=8),
        )

        boundaries = QubitSparsePauliList.from_sparse_list(
            [("XZ", (0, 1)), ("XX", (2, 3))], num_qubits=8
        )
        # Move a single-qubit term from the second summand into the first (the particular indices
        # ensure we remain term-wise sorted).
        boundaries.boundaries[1] += 1
        self.assertEqual(
            boundaries,
            QubitSparsePauliList.from_sparse_list([("XZX", (0, 1, 2)), ("X", (3,))], num_qubits=8),
        )

    def test_write_into_attributes_broadcast(self):
        # It's hard to broadcast into `indices` without breaking data coherence; the broadcasting is
        # more meant for fast modifications to `coeffs` and `paulis`.
        indices = QubitSparsePauliList.from_list(["XIIZI", "IIYIZ", "ZIIIY"])
        indices.indices[::2] = 1
        self.assertEqual(indices, QubitSparsePauliList.from_list(["XIIZI", "IIYZI", "ZIIYI"]))

        paulis = QubitSparsePauliList.from_list(["XIIZI", "IIYIZ", "ZIIIY"])
        paulis.paulis[::2] = QubitSparsePauliList.Pauli.Z
        self.assertEqual(
            paulis,
            QubitSparsePauliList.from_list(["XIIZI", "IIYIZ", "ZIIIZ"]),
        )
        paulis.paulis[3:1:-1] = QubitSparsePauliList.Pauli.X
        self.assertEqual(
            paulis,
            QubitSparsePauliList.from_list(["XIIZI", "IIXIX", "ZIIIZ"]),
        )
        paulis.paulis[paulis.boundaries[2] : paulis.boundaries[3]] = QubitSparsePauliList.Pauli.X
        self.assertEqual(
            paulis,
            QubitSparsePauliList.from_list(["XIIZI", "IIXIX", "XIIIX"]),
        )

        boundaries = QubitSparsePauliList.from_list(["IIIIZX", "IIXXII", "YYIIII"])
        boundaries.boundaries[1:3] = 1
        self.assertEqual(
            boundaries,
            QubitSparsePauliList.from_list(["IIIIIX", "IIIIII", "YYXXZI"]),
        )

    def test_write_into_attributes_slice(self):
        indices = QubitSparsePauliList.from_list(["IIIIZX", "IIXYII", "YZIIII"])
        indices.indices[:4] = [4, 5, 1, 2]
        self.assertEqual(indices, QubitSparsePauliList.from_list(["ZXIIII", "IIIXYI", "YZIIII"]))

        paulis = QubitSparsePauliList.from_list(["IIIIZX", "IIXXII", "YYIIII"])
        paulis.paulis[::2] = [
            QubitSparsePauliList.Pauli.Y,
            QubitSparsePauliList.Pauli.Y,
            QubitSparsePauliList.Pauli.Z,
        ]
        self.assertEqual(
            paulis,
            QubitSparsePauliList.from_list(["IIIIZY", "IIXYII", "YZIIII"]),
        )

        boundaries = QubitSparsePauliList.from_list(["IIIIZX", "IIXXII", "YYIIII"])
        boundaries.boundaries[1:-1] = [1, 5]
        self.assertEqual(
            boundaries,
            QubitSparsePauliList.from_list(["IIIIIX", "IYXXZI", "YIIIII"]),
        )

    def test_attributes_reject_bad_writes(self):
        pauli_list = QubitSparsePauliList.from_list(["XZY", "XXY"])
        with self.assertRaises(TypeError):
            pauli_list.paulis[0] = [QubitSparsePauliList.Pauli.X] * 4
        with self.assertRaises(TypeError):
            pauli_list.indices[0] = [0, 1]
        with self.assertRaises(TypeError):
            pauli_list.boundaries[0] = (0, 1)
        with self.assertRaisesRegex(ValueError, "not a valid letter"):
            pauli_list.paulis[0] = 0
        with self.assertRaisesRegex(ValueError, "not a valid letter"):
            pauli_list.paulis[:] = 0
        with self.assertRaisesRegex(
            ValueError, "tried to set a slice of length 6 with a sequence of length 8"
        ):
            pauli_list.paulis[:] = [QubitSparsePauliList.Pauli.Z] * 8

    def test_attributes_sequence(self):
        """Test attributes of the `Sequence` protocol."""
        # Length
        pauli_list = QubitSparsePauliList.from_list(["XZY", "ZYX"])
        self.assertEqual(len(pauli_list.indices), 6)
        self.assertEqual(len(pauli_list.paulis), 6)
        self.assertEqual(len(pauli_list.boundaries), 3)

        # Iteration
        self.assertEqual(tuple(pauli_list.indices), (0, 1, 2, 0, 1, 2))
        self.assertEqual(next(iter(pauli_list.boundaries)), 0)
        # multiple iteration through same object
        paulis = pauli_list.paulis
        self.assertEqual(set(paulis), {QubitSparsePauliList.Pauli[x] for x in "XYZZYX"})
        self.assertEqual(set(paulis), {QubitSparsePauliList.Pauli[x] for x in "XYZZYX"})

        # Implicit iteration methods.
        self.assertIn(QubitSparsePauliList.Pauli.Y, pauli_list.paulis)
        self.assertNotIn(4, pauli_list.indices)

        # Index by scalar
        self.assertEqual(pauli_list.indices[-1], 2)
        self.assertEqual(pauli_list.paulis[0], QubitSparsePauliList.Pauli.Y)
        # Make sure that Rust-space actually returns the enum value, not just an `int` (which could
        # have compared equal).
        self.assertIsInstance(pauli_list.paulis[0], QubitSparsePauliList.Pauli)
        self.assertEqual(pauli_list.boundaries[-2], 3)
        with self.assertRaises(IndexError):
            _ = pauli_list.boundaries[-4]

        # Index by slice.  This is API guaranteed to be a Numpy array to make it easier to
        # manipulate subslices with mathematic operations.
        self.assertIsInstance(pauli_list.indices[::-1], np.ndarray)
        np.testing.assert_array_equal(
            pauli_list.indices[::-1],
            np.array([2, 1, 0, 2, 1, 0], dtype=np.uint32),
            strict=True,
        )
        self.assertIsInstance(pauli_list.paulis[2:4], np.ndarray)
        np.testing.assert_array_equal(
            pauli_list.paulis[2:4],
            np.array([QubitSparsePauliList.Pauli.X, QubitSparsePauliList.Pauli.X], dtype=np.uint8),
            strict=True,
        )
        self.assertIsInstance(pauli_list.boundaries[-2:-3:-1], np.ndarray)
        np.testing.assert_array_equal(
            pauli_list.boundaries[-2:-3:-1], np.array([3], dtype=np.uintp), strict=True
        )

    def test_attributes_to_array(self):
        pauli_list = QubitSparsePauliList.from_list(["XZY", "XYZ"])

        # Natural dtypes.
        np.testing.assert_array_equal(
            pauli_list.indices, np.array([0, 1, 2, 0, 1, 2], dtype=np.uint32), strict=True
        )
        np.testing.assert_array_equal(
            pauli_list.paulis,
            np.array([QubitSparsePauliList.Pauli[x] for x in "YZXZYX"], dtype=np.uint8),
            strict=True,
        )
        np.testing.assert_array_equal(
            pauli_list.boundaries, np.array([0, 3, 6], dtype=np.uintp), strict=True
        )

        # Cast dtypes.
        np.testing.assert_array_equal(
            np.array(pauli_list.indices, dtype=np.uint8),
            np.array([0, 1, 2, 0, 1, 2], dtype=np.uint8),
            strict=True,
        )
        np.testing.assert_array_equal(
            np.array(pauli_list.boundaries, dtype=np.int64),
            np.array([0, 3, 6], dtype=np.int64),
            strict=True,
        )

    @unittest.skipIf(
        int(np.__version__.split(".", maxsplit=1)[0]) < 2,
        "Numpy 1.x did not have a 'copy' keyword parameter to 'numpy.asarray'",
    )
    def test_attributes_reject_no_copy_array(self):
        pauli_list = QubitSparsePauliList.from_list(["XZY", "YXZ"])
        with self.assertRaisesRegex(ValueError, "cannot produce a safe view"):
            np.asarray(pauli_list.indices, copy=False)
        with self.assertRaisesRegex(ValueError, "cannot produce a safe view"):
            np.asarray(pauli_list.paulis, copy=False)
        with self.assertRaisesRegex(ValueError, "cannot produce a safe view"):
            np.asarray(pauli_list.boundaries, copy=False)

    def test_attributes_repr(self):
        # We're not testing much about the outputs here, just that they don't crash.
        pauli_list = QubitSparsePauliList.from_list(["XZY", "YXZ"])
        self.assertIn("paulis", repr(pauli_list.paulis))
        self.assertIn("indices", repr(pauli_list.indices))
        self.assertIn("boundaries", repr(pauli_list.boundaries))

    @ddt.idata(single_cases_list())
    def test_clear(self, pauli_list):
        num_qubits = pauli_list.num_qubits
        pauli_list.clear()
        self.assertEqual(pauli_list, QubitSparsePauliList.empty(num_qubits))

    def test_iteration(self):
        self.assertEqual(list(QubitSparsePauliList.empty(5)), [])
        self.assertEqual(tuple(QubitSparsePauliList.empty(0)), ())

        pauli_list = QubitSparsePauliList.from_sparse_list(
            [
                ("XYY", (4, 2, 1)),
                ("", ()),
                ("ZZ", (3, 0)),
                ("XX", (2, 1)),
                ("YZ", (4, 1)),
            ],
            num_qubits=5,
        )
        pauli = QubitSparsePauliList.Pauli
        expected = [
            QubitSparsePauli.from_raw_parts(5, [pauli.Y, pauli.Y, pauli.X], [1, 2, 4]),
            QubitSparsePauli.from_raw_parts(5, [], []),
            QubitSparsePauli.from_raw_parts(5, [pauli.Z, pauli.Z], [0, 3]),
            QubitSparsePauli.from_raw_parts(5, [pauli.X, pauli.X], [1, 2]),
            QubitSparsePauli.from_raw_parts(5, [pauli.Z, pauli.Y], [1, 4]),
        ]
        self.assertEqual(list(pauli_list), expected)

    def test_indexing(self):
        pauli_list = QubitSparsePauliList.from_sparse_list(
            [
                ("XYY", (4, 2, 1)),
                ("", ()),
                ("ZZ", (3, 0)),
                ("XX", (2, 1)),
                ("YZ", (4, 1)),
            ],
            num_qubits=5,
        )
        pauli = QubitSparsePauliList.Pauli
        expected = [
            QubitSparsePauli.from_raw_parts(5, [pauli.Y, pauli.Y, pauli.X], [1, 2, 4]),
            QubitSparsePauli.from_raw_parts(5, [], []),
            QubitSparsePauli.from_raw_parts(5, [pauli.Z, pauli.Z], [0, 3]),
            QubitSparsePauli.from_raw_parts(5, [pauli.X, pauli.X], [1, 2]),
            QubitSparsePauli.from_raw_parts(5, [pauli.Z, pauli.Y], [1, 4]),
        ]
        self.assertEqual(pauli_list[0], expected[0])
        self.assertEqual(pauli_list[-2], expected[-2])
        self.assertEqual(pauli_list[2:4], QubitSparsePauliList(expected[2:4]))
        self.assertEqual(pauli_list[1::2], QubitSparsePauliList(expected[1::2]))
        self.assertEqual(pauli_list[:], QubitSparsePauliList(expected))
        self.assertEqual(pauli_list[-1:-4:-1], QubitSparsePauliList(expected[-1:-4:-1]))

    def test_to_sparse_list(self):
        """Test converting to a sparse list."""
        with self.subTest(msg="empty"):
            pauli_list = QubitSparsePauliList.empty(100)
            expected = []
            self.assertEqual(expected, pauli_list.to_sparse_list())

        with self.subTest(msg="IXYZ"):
            pauli_list = QubitSparsePauliList(["IXYZ"])
            expected = [("ZYX", [0, 1, 2])]
            self.assertEqual(
                canonicalize_sparse_list(expected),
                canonicalize_sparse_list(pauli_list.to_sparse_list()),
            )

        with self.subTest(msg="multiple"):
            pauli_list = QubitSparsePauliList.from_list(["XXIZ", "YYIZ"])
            expected = [("XXZ", [3, 2, 0]), ("ZYY", [0, 2, 3])]
            self.assertEqual(
                canonicalize_sparse_list(expected),
                canonicalize_sparse_list(pauli_list.to_sparse_list()),
            )


def canonicalize_term(pauli, indices):
    # canonicalize a sparse list term by sorting by indices (which is unique as
    # indices cannot be repeated)
    idcs = np.argsort(indices)
    sorted_paulis = "".join(pauli[i] for i in idcs)
    return (sorted_paulis, np.asarray(indices)[idcs].tolist())


def canonicalize_sparse_list(sparse_list):
    # sort a sparse list representation by canonicalizing the terms and then applying
    # Python's built-in sort
    canonicalized_terms = [canonicalize_term(*term) for term in sparse_list]
    return sorted(canonicalized_terms)
