# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
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

from qiskit.circuit import Parameter
from qiskit.quantum_info import SparseObservable, SparsePauliOp, Pauli

from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt.ddt
class TestSparseObservable(QiskitTestCase):
    def test_default_constructor_pauli(self):
        data = Pauli("IXYIZ")
        self.assertEqual(SparseObservable(data), SparseObservable.from_pauli(data))
        self.assertEqual(
            SparseObservable(data, num_qubits=data.num_qubits), SparseObservable.from_pauli(data)
        )
        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            SparseObservable(data, num_qubits=data.num_qubits + 1)

        with_phase = Pauli("-jIYYXY")
        self.assertEqual(SparseObservable(with_phase), SparseObservable.from_pauli(with_phase))
        self.assertEqual(
            SparseObservable(with_phase, num_qubits=data.num_qubits),
            SparseObservable.from_pauli(with_phase),
        )

        self.assertEqual(SparseObservable(Pauli("")), SparseObservable.from_pauli(Pauli("")))

    def test_default_constructor_sparse_pauli_op(self):
        data = SparsePauliOp.from_list([("IIXIY", 1.0), ("XYYZI", -0.25), ("XYIYY", -0.25 + 0.75j)])
        self.assertEqual(SparseObservable(data), SparseObservable.from_sparse_pauli_op(data))
        self.assertEqual(
            SparseObservable(data, num_qubits=data.num_qubits),
            SparseObservable.from_sparse_pauli_op(data),
        )
        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            SparseObservable(data, num_qubits=data.num_qubits + 1)
        with self.assertRaisesRegex(TypeError, "complex-typed coefficients"):
            SparseObservable(SparsePauliOp(["XX"], [Parameter("x")]))

    def test_default_constructor_label(self):
        data = "IIXI+-I01rlIYZ"
        self.assertEqual(SparseObservable(data), SparseObservable.from_label(data))
        self.assertEqual(
            SparseObservable(data, num_qubits=len(data)), SparseObservable.from_label(data)
        )
        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            SparseObservable(data, num_qubits=len(data) + 1)

    def test_default_constructor_list(self):
        data = [("IXIIZ", 0.5), ("+I-II", 1.0 - 0.25j), ("IIrlI", -0.75)]
        self.assertEqual(SparseObservable(data), SparseObservable.from_list(data))
        self.assertEqual(SparseObservable(data, num_qubits=5), SparseObservable.from_list(data))
        with self.assertRaisesRegex(ValueError, "label with length 5 cannot be added"):
            SparseObservable(data, num_qubits=4)
        with self.assertRaisesRegex(ValueError, "label with length 5 cannot be added"):
            SparseObservable(data, num_qubits=6)
        self.assertEqual(
            SparseObservable([], num_qubits=5), SparseObservable.from_list([], num_qubits=5)
        )

    def test_default_constructor_sparse_list(self):
        data = [("ZX", (0, 3), 0.5), ("-+", (2, 4), 1.0 - 0.25j), ("rl", (2, 1), -0.75)]
        self.assertEqual(
            SparseObservable(data, num_qubits=5),
            SparseObservable.from_sparse_list(data, num_qubits=5),
        )
        self.assertEqual(
            SparseObservable(data, num_qubits=10),
            SparseObservable.from_sparse_list(data, num_qubits=10),
        )
        with self.assertRaisesRegex(ValueError, "'num_qubits' must be provided"):
            SparseObservable(data)
        self.assertEqual(
            SparseObservable([], num_qubits=5), SparseObservable.from_sparse_list([], num_qubits=5)
        )

    def test_default_constructor_copy(self):
        base = SparseObservable.from_list([("IXIZIY", 1.0), ("+-rl01", -1.0j)])
        copied = SparseObservable(base)
        self.assertEqual(base, copied)
        self.assertIsNot(base, copied)

        # Modifications to `copied` don't propagate back.
        copied.coeffs[1] = -0.5j
        self.assertNotEqual(base, copied)

        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            SparseObservable(base, num_qubits=base.num_qubits + 1)

    def test_default_constructor_failed_inference(self):
        with self.assertRaises(TypeError):
            # Mixed dense/sparse list.
            SparseObservable([("IIXIZ", 1.0), ("+-", (2, 3), -1.0)], num_qubits=5)

    def test_num_qubits(self):
        self.assertEqual(SparseObservable.zero(0).num_qubits, 0)
        self.assertEqual(SparseObservable.zero(10).num_qubits, 10)

        self.assertEqual(SparseObservable.identity(0).num_qubits, 0)
        self.assertEqual(SparseObservable.identity(1_000_000).num_qubits, 1_000_000)

    def test_num_terms(self):
        self.assertEqual(SparseObservable.zero(0).num_terms, 0)
        self.assertEqual(SparseObservable.zero(10).num_terms, 0)
        self.assertEqual(SparseObservable.identity(0).num_terms, 1)
        self.assertEqual(SparseObservable.identity(1_000_000).num_terms, 1)
        self.assertEqual(
            SparseObservable.from_list([("IIIXIZ", 1.0), ("YY+-II", 0.5j)]).num_terms, 2
        )

    def test_bit_term_enum(self):
        # These are very explicit tests that effectively just duplicate magic numbers, but the point
        # is that those magic numbers are required to be constant as their values are part of the
        # public interface.

        self.assertEqual(
            set(SparseObservable.BitTerm),
            {
                SparseObservable.BitTerm.X,
                SparseObservable.BitTerm.Y,
                SparseObservable.BitTerm.Z,
                SparseObservable.BitTerm.PLUS,
                SparseObservable.BitTerm.MINUS,
                SparseObservable.BitTerm.RIGHT,
                SparseObservable.BitTerm.LEFT,
                SparseObservable.BitTerm.ZERO,
                SparseObservable.BitTerm.ONE,
            },
        )
        # All the enumeration items should also be integers.
        self.assertIsInstance(SparseObservable.BitTerm.X, int)
        values = {
            "X": 0b00_10,
            "Y": 0b00_11,
            "Z": 0b00_01,
            "PLUS": 0b10_10,
            "MINUS": 0b01_10,
            "RIGHT": 0b10_11,
            "LEFT": 0b01_11,
            "ZERO": 0b10_01,
            "ONE": 0b01_01,
        }
        self.assertEqual({name: getattr(SparseObservable.BitTerm, name) for name in values}, values)

        # The single-character label aliases can be accessed with index notation.
        labels = {
            "X": SparseObservable.BitTerm.X,
            "Y": SparseObservable.BitTerm.Y,
            "Z": SparseObservable.BitTerm.Z,
            "+": SparseObservable.BitTerm.PLUS,
            "-": SparseObservable.BitTerm.MINUS,
            "r": SparseObservable.BitTerm.RIGHT,
            "l": SparseObservable.BitTerm.LEFT,
            "0": SparseObservable.BitTerm.ZERO,
            "1": SparseObservable.BitTerm.ONE,
        }
        self.assertEqual({label: SparseObservable.BitTerm[label] for label in labels}, labels)

    @ddt.data(
        SparseObservable.zero(0),
        SparseObservable.zero(10),
        SparseObservable.identity(0),
        SparseObservable.identity(1_000),
        SparseObservable.from_label("IIXIZI"),
        SparseObservable.from_list([("YIXZII", -0.25), ("01rl+-", 0.25 + 0.5j)]),
    )
    def test_pickle(self, observable):
        self.assertEqual(observable, copy.copy(observable))
        self.assertIsNot(observable, copy.copy(observable))
        self.assertEqual(observable, copy.deepcopy(observable))
        self.assertEqual(observable, pickle.loads(pickle.dumps(observable)))

    @ddt.data(
        # This is every combination of (0, 1, many) for (terms, qubits, non-identites per term).
        SparseObservable.zero(0),
        SparseObservable.zero(1),
        SparseObservable.zero(10),
        SparseObservable.identity(0),
        SparseObservable.identity(1),
        SparseObservable.identity(1_000),
        SparseObservable.from_label("IIXIZI"),
        SparseObservable.from_label("X"),
        SparseObservable.from_list([("YIXZII", -0.25), ("01rl+-", 0.25 + 0.5j)]),
    )
    def test_repr(self, data):
        # The purpose of this is just to test that the `repr` doesn't crash, rather than asserting
        # that it has any particular form.
        self.assertIsInstance(repr(data), str)
        self.assertIn("SparseObservable", repr(data))

    def test_from_raw_parts(self):
        # Happiest path: exactly typed inputs.
        num_qubits = 100
        terms = np.full((num_qubits,), SparseObservable.BitTerm.Z, dtype=np.uint8)
        indices = np.arange(num_qubits, dtype=np.uint32)
        coeffs = np.ones((num_qubits,), dtype=complex)
        boundaries = np.arange(num_qubits + 1, dtype=np.uintp)
        observable = SparseObservable.from_raw_parts(num_qubits, coeffs, terms, indices, boundaries)
        self.assertEqual(observable.num_qubits, num_qubits)
        np.testing.assert_equal(observable.bit_terms, terms)
        np.testing.assert_equal(observable.indices, indices)
        np.testing.assert_equal(observable.coeffs, coeffs)
        np.testing.assert_equal(observable.boundaries, boundaries)

        self.assertEqual(
            observable,
            SparseObservable.from_raw_parts(
                num_qubits, coeffs, terms, indices, boundaries, check=False
            ),
        )

        # At least the initial implementation of `SparseObservable` requires `from_raw_parts` to be
        # a copy constructor in order to allow it to be resized by Rust space.  This is checking for
        # that, but if the implementation changes, it could potentially be relaxed.
        self.assertFalse(np.may_share_memory(observable.coeffs, coeffs))

        # Conversion from array-likes, including mis-typed but compatible arrays.
        observable = SparseObservable.from_raw_parts(
            num_qubits, list(coeffs), tuple(terms), observable.indices, boundaries.astype(np.int16)
        )
        self.assertEqual(observable.num_qubits, num_qubits)
        np.testing.assert_equal(observable.bit_terms, terms)
        np.testing.assert_equal(observable.indices, indices)
        np.testing.assert_equal(observable.coeffs, coeffs)
        np.testing.assert_equal(observable.boundaries, boundaries)

        # Construction of zero operator.
        self.assertEqual(
            SparseObservable.from_raw_parts(10, [], [], [], [0]), SparseObservable.zero(10)
        )

        # Construction of an operator with an intermediate identity term.  For the initial
        # constructor tests, it's hard to check anything more than the construction succeeded.
        self.assertEqual(
            SparseObservable.from_raw_parts(
                10, [1.0j, 0.5, 2.0], [1, 3, 2], [0, 1, 2], [0, 1, 1, 3]
            ).num_terms,
            # The three are [(1.0j)*(Z_1), 0.5, 2.0*(X_2 Y_1)]
            3,
        )

    def test_from_raw_parts_checks_coherence(self):
        with self.assertRaisesRegex(ValueError, "not a valid letter"):
            SparseObservable.from_raw_parts(2, [1.0j], [ord("$")], [0], [0, 1])
        with self.assertRaisesRegex(ValueError, r"boundaries.*must be one element longer"):
            SparseObservable.from_raw_parts(2, [1.0j], [SparseObservable.BitTerm.Z], [0], [0])
        with self.assertRaisesRegex(ValueError, r"`bit_terms` \(1\) and `indices` \(0\)"):
            SparseObservable.from_raw_parts(2, [1.0j], [SparseObservable.BitTerm.Z], [], [0, 1])
        with self.assertRaisesRegex(ValueError, r"`bit_terms` \(0\) and `indices` \(1\)"):
            SparseObservable.from_raw_parts(2, [1.0j], [], [1], [0, 1])
        with self.assertRaisesRegex(ValueError, r"the first item of `boundaries` \(1\) must be 0"):
            SparseObservable.from_raw_parts(2, [1.0j], [SparseObservable.BitTerm.Z], [0], [1, 1])
        with self.assertRaisesRegex(ValueError, r"the last item of `boundaries` \(2\)"):
            SparseObservable.from_raw_parts(2, [1.0j], [1], [0], [0, 2])
        with self.assertRaisesRegex(ValueError, r"the last item of `boundaries` \(1\)"):
            SparseObservable.from_raw_parts(2, [1.0j], [1, 2], [0, 1], [0, 1])
        with self.assertRaisesRegex(ValueError, r"all qubit indices must be less than the number"):
            SparseObservable.from_raw_parts(4, [1.0j], [1, 2], [0, 4], [0, 2])
        with self.assertRaisesRegex(ValueError, r"all qubit indices must be less than the number"):
            SparseObservable.from_raw_parts(4, [1.0j, -0.5j], [1, 2], [0, 4], [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "the values in `boundaries` include backwards"):
            SparseObservable.from_raw_parts(
                5, [1.0j, -0.5j, 2.0], [1, 2, 3, 2], [0, 1, 2, 3], [0, 2, 1, 4]
            )
        with self.assertRaisesRegex(
            ValueError, "the values in `indices` are not term-wise increasing"
        ):
            SparseObservable.from_raw_parts(4, [1.0j], [1, 2], [1, 0], [0, 2])

        # There's no test of attempting to pass incoherent data and `check=False` because that
        # permits undefined behaviour in Rust (it's unsafe), so all bets would be off.

    def test_from_label(self):
        # The label is interpreted like a bitstring, with the right-most item associated with qubit
        # 0, and increasing as we move to the left (like `Pauli`, and other bitstring conventions).
        self.assertEqual(
            # Ruler for counting terms:  dcba9876543210
            SparseObservable.from_label("I+-II01IrlIXYZ"),
            SparseObservable.from_raw_parts(
                14,
                [1.0],
                [
                    SparseObservable.BitTerm.Z,
                    SparseObservable.BitTerm.Y,
                    SparseObservable.BitTerm.X,
                    SparseObservable.BitTerm.LEFT,
                    SparseObservable.BitTerm.RIGHT,
                    SparseObservable.BitTerm.ONE,
                    SparseObservable.BitTerm.ZERO,
                    SparseObservable.BitTerm.MINUS,
                    SparseObservable.BitTerm.PLUS,
                ],
                [0, 1, 2, 4, 5, 7, 8, 11, 12],
                [0, 9],
            ),
        )

        self.assertEqual(SparseObservable.from_label("I" * 10), SparseObservable.identity(10))

        # The empty label case is a 0-qubit identity, since `from_label` always sets a coefficient
        # of 1.
        self.assertEqual(SparseObservable.from_label(""), SparseObservable.identity(0))

    def test_from_label_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            SparseObservable.from_label("I+-$%I")
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Unicode shenangigans.
            SparseObservable.from_label("🐍")

    def test_from_list(self):
        label = "IXYI+-0lr1ZZY"
        self.assertEqual(
            SparseObservable.from_list([(label, 1.0)]), SparseObservable.from_label(label)
        )
        self.assertEqual(
            SparseObservable.from_list([(label, 1.0)], num_qubits=len(label)),
            SparseObservable.from_label(label),
        )

        self.assertEqual(
            SparseObservable.from_list([("IIIXZI", 1.0j), ("+-IIII", -0.5)]),
            SparseObservable.from_raw_parts(
                6,
                [1.0j, -0.5],
                [
                    SparseObservable.BitTerm.Z,
                    SparseObservable.BitTerm.X,
                    SparseObservable.BitTerm.MINUS,
                    SparseObservable.BitTerm.PLUS,
                ],
                [1, 2, 4, 5],
                [0, 2, 4],
            ),
        )

        self.assertEqual(SparseObservable.from_list([], num_qubits=5), SparseObservable.zero(5))
        self.assertEqual(SparseObservable.from_list([], num_qubits=0), SparseObservable.zero(0))

    def test_from_list_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            SparseObservable.from_list([("XZIIZY", 0.5), ("I+-$%I", 1.0j)])
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Unicode shenangigans.
            SparseObservable.from_list([("🐍", 0.5)])
        with self.assertRaisesRegex(ValueError, "label with length 4 cannot be added"):
            SparseObservable.from_list([("IIZ", 0.5), ("IIXI", 1.0j)])
        with self.assertRaisesRegex(ValueError, "label with length 2 cannot be added"):
            SparseObservable.from_list([("IIZ", 0.5), ("II", 1.0j)])
        with self.assertRaisesRegex(ValueError, "label with length 3 cannot be added"):
            SparseObservable.from_list([("IIZ", 0.5), ("IXI", 1.0j)], num_qubits=2)
        with self.assertRaisesRegex(ValueError, "label with length 3 cannot be added"):
            SparseObservable.from_list([("IIZ", 0.5), ("IXI", 1.0j)], num_qubits=4)
        with self.assertRaisesRegex(ValueError, "cannot construct.*without knowing `num_qubits`"):
            SparseObservable.from_list([])

    def test_from_sparse_list(self):
        self.assertEqual(
            SparseObservable.from_sparse_list(
                [
                    ("XY", (0, 1), 0.5),
                    ("+-", (1, 3), -0.25j),
                    ("rl0", (0, 2, 4), 1.0j),
                ],
                num_qubits=5,
            ),
            SparseObservable.from_list([("IIIYX", 0.5), ("I-I+I", -0.25j), ("0IlIr", 1.0j)]),
        )

        # The indices should be allowed to be given in unsorted order, but they should be term-wise
        # sorted in the output.
        from_unsorted = SparseObservable.from_sparse_list(
            [
                ("XYZ", (2, 1, 0), 1.5j),
                ("+rl", (2, 0, 1), -0.5),
            ],
            num_qubits=3,
        )
        self.assertEqual(from_unsorted, SparseObservable.from_list([("XYZ", 1.5j), ("+lr", -0.5)]))
        np.testing.assert_equal(
            from_unsorted.indices, np.array([0, 1, 2, 0, 1, 2], dtype=np.uint32)
        )

        # Explicit identities should still work, just be skipped over.
        explicit_identity = SparseObservable.from_sparse_list(
            [
                ("ZXI", (0, 1, 2), 1.0j),
                ("XYIII", (0, 1, 2, 3, 8), -0.5j),
            ],
            num_qubits=10,
        )
        self.assertEqual(
            explicit_identity,
            SparseObservable.from_sparse_list(
                [("XZ", (1, 0), 1.0j), ("YX", (1, 0), -0.5j)], num_qubits=10
            ),
        )
        np.testing.assert_equal(explicit_identity.indices, np.array([0, 1, 0, 1], dtype=np.uint32))

        self.assertEqual(
            SparseObservable.from_sparse_list([("", (), 1.0)], num_qubits=5),
            SparseObservable.identity(5),
        )
        self.assertEqual(
            SparseObservable.from_sparse_list([("", (), 1.0)], num_qubits=0),
            SparseObservable.identity(0),
        )

        self.assertEqual(
            SparseObservable.from_sparse_list([], num_qubits=1_000_000),
            SparseObservable.zero(1_000_000),
        )
        self.assertEqual(
            SparseObservable.from_sparse_list([], num_qubits=0),
            SparseObservable.zero(0),
        )

    def test_from_sparse_list_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            SparseObservable.from_sparse_list(
                [("XZZY", (5, 3, 1, 0), 0.5), ("+$", (2, 1), 1.0j)], num_qubits=8
            )
        # Unicode shenangigans.  These two should fail with a `ValueError`, but the exact message
        # isn't important.  "\xff" is "ÿ", which is two bytes in UTF-8 (so has a length of 2 in
        # Rust), but has a length of 1 in Python, so try with both a length-1 and length-2 index
        # sequence, and both should still raise `ValueError`.
        with self.assertRaises(ValueError):
            SparseObservable.from_sparse_list([("\xff", (1,), 0.5)], num_qubits=5)
        with self.assertRaises(ValueError):
            SparseObservable.from_sparse_list([("\xff", (1, 2), 0.5)], num_qubits=5)

        with self.assertRaisesRegex(ValueError, "label with length 2 does not match indices"):
            SparseObservable.from_sparse_list([("XZ", (0,), 1.0)], num_qubits=5)
        with self.assertRaisesRegex(ValueError, "label with length 2 does not match indices"):
            SparseObservable.from_sparse_list([("XZ", (0, 1, 2), 1.0)], num_qubits=5)

        with self.assertRaisesRegex(ValueError, "index 3 is out of range for a 3-qubit operator"):
            SparseObservable.from_sparse_list([("XZY", (0, 1, 3), 1.0)], num_qubits=3)
        with self.assertRaisesRegex(ValueError, "index 4 is out of range for a 3-qubit operator"):
            SparseObservable.from_sparse_list([("XZY", (0, 1, 4), 1.0)], num_qubits=3)
        with self.assertRaisesRegex(ValueError, "index 3 is out of range for a 3-qubit operator"):
            # ... even if it's for an explicit identity.
            SparseObservable.from_sparse_list([("+-I", (0, 1, 3), 1.0)], num_qubits=3)

        with self.assertRaisesRegex(ValueError, "index 3 is duplicated"):
            SparseObservable.from_sparse_list([("XZ", (3, 3), 1.0)], num_qubits=5)
        with self.assertRaisesRegex(ValueError, "index 3 is duplicated"):
            SparseObservable.from_sparse_list([("XYZXZ", (3, 0, 1, 2, 3), 1.0)], num_qubits=5)

    def test_from_pauli(self):
        # This function should be infallible provided `Pauli` doesn't change its interface and the
        # user doesn't violate the typing.

        # Simple check that the labels are interpreted in the same order.
        self.assertEqual(
            SparseObservable.from_pauli(Pauli("IIXZI")), SparseObservable.from_label("IIXZI")
        )

        # `Pauli` accepts a phase in its label, which we can't (because of clashes with the other
        # alphabet letters), and we should get that right.
        self.assertEqual(
            SparseObservable.from_pauli(Pauli("iIXZIX")),
            SparseObservable.from_list([("IXZIX", 1.0j)]),
        )
        self.assertEqual(
            SparseObservable.from_pauli(Pauli("-iIXZIX")),
            SparseObservable.from_list([("IXZIX", -1.0j)]),
        )
        self.assertEqual(
            SparseObservable.from_pauli(Pauli("-IXZIX")),
            SparseObservable.from_list([("IXZIX", -1.0)]),
        )

        # `Pauli` has its internal phase convention for how it stores `Y`; we should get this right
        # regardless of how many Ys are in the label, or if there's a phase.
        paulis = {"IXYZ" * n: Pauli("IXYZ" * n) for n in range(1, 5)}
        from_paulis, from_labels = zip(
            *(
                (SparseObservable.from_pauli(pauli), SparseObservable.from_label(label))
                for label, pauli in paulis.items()
            )
        )
        self.assertEqual(from_paulis, from_labels)

        phased_paulis = {"IXYZ" * n: Pauli("j" + "IXYZ" * n) for n in range(1, 5)}
        from_paulis, from_lists = zip(
            *(
                (SparseObservable.from_pauli(pauli), SparseObservable.from_list([(label, 1.0j)]))
                for label, pauli in phased_paulis.items()
            )
        )
        self.assertEqual(from_paulis, from_lists)

        self.assertEqual(SparseObservable.from_pauli(Pauli("III")), SparseObservable.identity(3))
        self.assertEqual(SparseObservable.from_pauli(Pauli("")), SparseObservable.identity(0))

    def test_from_sparse_pauli_op(self):
        self.assertEqual(
            SparseObservable.from_sparse_pauli_op(SparsePauliOp.from_list([("IIIII", 1.0)])),
            SparseObservable.identity(5),
        )

        data = [("ZXZXZ", 0.25), ("IYXZI", 1.0j), ("IYYZX", 0.5), ("YYYXI", -0.5), ("IYYYY", 2j)]
        self.assertEqual(
            SparseObservable.from_sparse_pauli_op(SparsePauliOp.from_list(data)),
            SparseObservable.from_list(data),
        )

        # These two _should_ produce the same structure as `SparseObservable.zero(num_qubits)`, but
        # because `SparsePauliOp` doesn't represent the zero operator "natively" - with an empty sum
        # - they actually come out looking like `0.0` times the identity, which is less efficient
        # but acceptable.
        self.assertEqual(
            SparseObservable.from_sparse_pauli_op(SparsePauliOp.from_list([], num_qubits=1)),
            SparseObservable.from_list([("I", 0.0)]),
        )
        self.assertEqual(
            SparseObservable.from_sparse_pauli_op(SparsePauliOp.from_list([], num_qubits=0)),
            SparseObservable.from_list([("", 0.0)]),
        )

    def test_from_sparse_pauli_op_failures(self):
        parametric = SparsePauliOp.from_list([("IIXZ", Parameter("x"))], dtype=object)
        with self.assertRaisesRegex(TypeError, "complex-typed coefficients"):
            SparseObservable.from_sparse_pauli_op(parametric)

    def test_zero(self):
        zero_5 = SparseObservable.zero(5)
        self.assertEqual(zero_5.num_qubits, 5)
        np.testing.assert_equal(zero_5.coeffs, np.array([], dtype=complex))
        np.testing.assert_equal(zero_5.bit_terms, np.array([], dtype=np.uint8))
        np.testing.assert_equal(zero_5.indices, np.array([], dtype=np.uint32))
        np.testing.assert_equal(zero_5.boundaries, np.array([0], dtype=np.uintp))

        zero_0 = SparseObservable.zero(0)
        self.assertEqual(zero_0.num_qubits, 0)
        np.testing.assert_equal(zero_0.coeffs, np.array([], dtype=complex))
        np.testing.assert_equal(zero_0.bit_terms, np.array([], dtype=np.uint8))
        np.testing.assert_equal(zero_0.indices, np.array([], dtype=np.uint32))
        np.testing.assert_equal(zero_0.boundaries, np.array([0], dtype=np.uintp))

    def test_identity(self):
        id_5 = SparseObservable.identity(5)
        self.assertEqual(id_5.num_qubits, 5)
        np.testing.assert_equal(id_5.coeffs, np.array([1], dtype=complex))
        np.testing.assert_equal(id_5.bit_terms, np.array([], dtype=np.uint8))
        np.testing.assert_equal(id_5.indices, np.array([], dtype=np.uint32))
        np.testing.assert_equal(id_5.boundaries, np.array([0, 0], dtype=np.uintp))

        id_0 = SparseObservable.identity(0)
        self.assertEqual(id_0.num_qubits, 0)
        np.testing.assert_equal(id_0.coeffs, np.array([1], dtype=complex))
        np.testing.assert_equal(id_0.bit_terms, np.array([], dtype=np.uint8))
        np.testing.assert_equal(id_0.indices, np.array([], dtype=np.uint32))
        np.testing.assert_equal(id_0.boundaries, np.array([0, 0], dtype=np.uintp))

    @ddt.data(
        SparseObservable.zero(0),
        SparseObservable.zero(5),
        SparseObservable.identity(0),
        SparseObservable.identity(1_000_000),
        SparseObservable.from_list([("+-rl01", -0.5), ("IXZYZI", 1.0j)]),
    )
    def test_copy(self, obs):
        self.assertEqual(obs, obs.copy())
        self.assertIsNot(obs, obs.copy())

    def test_equality(self):
        sparse_data = [("XZ", (1, 0), 0.5j), ("+lr", (3, 1, 0), -0.25)]
        op = SparseObservable.from_sparse_list(sparse_data, num_qubits=5)
        self.assertEqual(op, op.copy())
        # Take care that Rust space allows multiple views onto the same object.
        self.assertEqual(op, op)

        # Comparison to some other object shouldn't fail.
        self.assertNotEqual(op, None)

        # No costly automatic simplification (mathematically, these operators _are_ the same).
        self.assertNotEqual(
            SparseObservable.from_list([("+", 1.0), ("-", 1.0)]), SparseObservable.from_label("X")
        )

        # Difference in qubit count.
        self.assertNotEqual(
            op, SparseObservable.from_sparse_list(sparse_data, num_qubits=op.num_qubits + 1)
        )
        self.assertNotEqual(SparseObservable.zero(2), SparseObservable.zero(3))
        self.assertNotEqual(SparseObservable.identity(2), SparseObservable.identity(3))

        # Difference in coeffs.
        self.assertNotEqual(
            SparseObservable.from_list([("IIXZI", 1.0), ("+-rl0", -0.5j)]),
            SparseObservable.from_list([("IIXZI", 1.0), ("+-rl0", 0.5j)]),
        )
        self.assertNotEqual(
            SparseObservable.from_list([("IIXZI", 1.0), ("+-rl0", -0.5j)]),
            SparseObservable.from_list([("IIXZI", 1.0j), ("+-rl0", -0.5j)]),
        )

        # Difference in bit terms.
        self.assertNotEqual(
            SparseObservable.from_list([("IIXZI", 1.0), ("+-rl0", -0.5j)]),
            SparseObservable.from_list([("IIYZI", 1.0), ("+-rl0", -0.5j)]),
        )
        self.assertNotEqual(
            SparseObservable.from_list([("IIXZI", 1.0), ("+-rl0", -0.5j)]),
            SparseObservable.from_list([("IIXZI", 1.0), ("+-rl1", -0.5j)]),
        )

        # Difference in indices.
        self.assertNotEqual(
            SparseObservable.from_list([("IIXZI", 1.0), ("+Irl0", -0.5j)]),
            SparseObservable.from_list([("IXIZI", 1.0), ("+Irl0", -0.5j)]),
        )
        self.assertNotEqual(
            SparseObservable.from_list([("IIXZI", 1.0), ("+Irl0", -0.5j)]),
            SparseObservable.from_list([("IIXZI", 1.0), ("I+rl0", -0.5j)]),
        )

        # Difference in boundaries.
        self.assertNotEqual(
            SparseObservable.from_sparse_list(
                [("XZ", (0, 1), 1.5), ("+-", (2, 3), -0.5j)], num_qubits=5
            ),
            SparseObservable.from_sparse_list(
                [("XZ+", (0, 1, 2), 1.5), ("-", (3,), -0.5j)], num_qubits=5
            ),
        )

    def test_write_into_attributes_scalar(self):
        coeffs = SparseObservable.from_sparse_list(
            [("XZ", (1, 0), 1.5j), ("+-", (3, 2), -1.5j)], num_qubits=8
        )
        coeffs.coeffs[0] = -2.0
        self.assertEqual(
            coeffs,
            SparseObservable.from_sparse_list(
                [("XZ", (1, 0), -2.0), ("+-", (3, 2), -1.5j)], num_qubits=8
            ),
        )
        coeffs.coeffs[1] = 1.5 + 0.25j
        self.assertEqual(
            coeffs,
            SparseObservable.from_sparse_list(
                [("XZ", (1, 0), -2.0), ("+-", (3, 2), 1.5 + 0.25j)], num_qubits=8
            ),
        )

        bit_terms = SparseObservable.from_sparse_list(
            [("XZ", (0, 1), 1.5j), ("+-", (2, 3), -1.5j)], num_qubits=8
        )
        bit_terms.bit_terms[0] = SparseObservable.BitTerm.Y
        bit_terms.bit_terms[3] = SparseObservable.BitTerm.LEFT
        self.assertEqual(
            bit_terms,
            SparseObservable.from_sparse_list(
                [("YZ", (0, 1), 1.5j), ("+l", (2, 3), -1.5j)], num_qubits=8
            ),
        )

        indices = SparseObservable.from_sparse_list(
            [("XZ", (0, 1), 1.5j), ("+-", (2, 3), -1.5j)], num_qubits=8
        )
        # These two sets keep the observable in term-wise increasing order.  We don't test what
        # happens if somebody violates the Rust-space requirement to be term-wise increasing.
        indices.indices[1] = 4
        indices.indices[3] = 7
        self.assertEqual(
            indices,
            SparseObservable.from_sparse_list(
                [("XZ", (0, 4), 1.5j), ("+-", (2, 7), -1.5j)], num_qubits=8
            ),
        )

        boundaries = SparseObservable.from_sparse_list(
            [("XZ", (0, 1), 1.5j), ("+-", (2, 3), -1.5j)], num_qubits=8
        )
        # Move a single-qubit term from the second summand into the first (the particular indices
        # ensure we remain term-wise sorted).
        boundaries.boundaries[1] += 1
        self.assertEqual(
            boundaries,
            SparseObservable.from_sparse_list(
                [("XZ+", (0, 1, 2), 1.5j), ("-", (3,), -1.5j)], num_qubits=8
            ),
        )

    def test_write_into_attributes_broadcast(self):
        coeffs = SparseObservable.from_list([("XIIZI", 1.5j), ("IIIl0", -0.25), ("1IIIY", 0.5)])
        coeffs.coeffs[:] = 1.5j
        np.testing.assert_array_equal(coeffs.coeffs, [1.5j, 1.5j, 1.5j])
        coeffs.coeffs[1:] = 1.0j
        np.testing.assert_array_equal(coeffs.coeffs, [1.5j, 1.0j, 1.0j])
        coeffs.coeffs[:2] = -0.5
        np.testing.assert_array_equal(coeffs.coeffs, [-0.5, -0.5, 1.0j])
        coeffs.coeffs[::2] = 1.5j
        np.testing.assert_array_equal(coeffs.coeffs, [1.5j, -0.5, 1.5j])
        coeffs.coeffs[::-1] = -0.5j
        np.testing.assert_array_equal(coeffs.coeffs, [-0.5j, -0.5j, -0.5j])

        # It's hard to broadcast into `indices` without breaking data coherence; the broadcasting is
        # more meant for fast modifications to `coeffs` and `bit_terms`.
        indices = SparseObservable.from_list([("XIIZI", 1.5j), ("IIlI0", -0.25), ("1IIIY", 0.5)])
        indices.indices[::2] = 1
        self.assertEqual(
            indices, SparseObservable.from_list([("XIIZI", 1.5j), ("IIl0I", -0.25), ("1IIYI", 0.5)])
        )

        bit_terms = SparseObservable.from_list([("XIIZI", 1.5j), ("IIlI0", -0.25), ("1IIIY", 0.5)])
        bit_terms.bit_terms[::2] = SparseObservable.BitTerm.Z
        self.assertEqual(
            bit_terms,
            SparseObservable.from_list([("XIIZI", 1.5j), ("IIlIZ", -0.25), ("1IIIZ", 0.5)]),
        )
        bit_terms.bit_terms[3:1:-1] = SparseObservable.BitTerm.PLUS
        self.assertEqual(
            bit_terms,
            SparseObservable.from_list([("XIIZI", 1.5j), ("II+I+", -0.25), ("1IIIZ", 0.5)]),
        )
        bit_terms.bit_terms[bit_terms.boundaries[2] : bit_terms.boundaries[3]] = (
            SparseObservable.BitTerm.MINUS
        )
        self.assertEqual(
            bit_terms,
            SparseObservable.from_list([("XIIZI", 1.5j), ("II+I+", -0.25), ("-III-", 0.5)]),
        )

        boundaries = SparseObservable.from_list([("IIIIZX", 1j), ("II+-II", -0.5), ("rlIIII", 0.5)])
        boundaries.boundaries[1:3] = 1
        self.assertEqual(
            boundaries,
            SparseObservable.from_list([("IIIIIX", 1j), ("IIIIII", -0.5), ("rl+-ZI", 0.5)]),
        )

    def test_write_into_attributes_slice(self):
        coeffs = SparseObservable.from_list([("XIIZI", 1.5j), ("IIIl0", -0.25), ("1IIIY", 0.5)])
        coeffs.coeffs[:] = [2.0, 0.5, -0.25]
        self.assertEqual(
            coeffs, SparseObservable.from_list([("XIIZI", 2.0), ("IIIl0", 0.5), ("1IIIY", -0.25)])
        )
        # This should assign the coefficients in reverse order - we more usually spell it
        # `coeffs[:] = coeffs{::-1]`, but the idea is to check the set-item slicing order.
        coeffs.coeffs[::-1] = coeffs.coeffs[:]
        self.assertEqual(
            coeffs, SparseObservable.from_list([("XIIZI", -0.25), ("IIIl0", 0.5), ("1IIIY", 2.0)])
        )

        indices = SparseObservable.from_list([("IIIIZX", 0.25), ("II+-II", 1j), ("rlIIII", 0.5)])
        indices.indices[:4] = [4, 5, 1, 2]
        self.assertEqual(
            indices, SparseObservable.from_list([("ZXIIII", 0.25), ("III+-I", 1j), ("rlIIII", 0.5)])
        )

        bit_terms = SparseObservable.from_list([("IIIIZX", 0.25), ("II+-II", 1j), ("rlIIII", 0.5)])
        bit_terms.bit_terms[::2] = [
            SparseObservable.BitTerm.Y,
            SparseObservable.BitTerm.RIGHT,
            SparseObservable.BitTerm.ZERO,
        ]
        self.assertEqual(
            bit_terms,
            SparseObservable.from_list([("IIIIZY", 0.25), ("II+rII", 1j), ("r0IIII", 0.5)]),
        )

        operators = SparseObservable.from_list([("XZY", 1.5j), ("+1r", -0.5)])
        # Reduce all single-qubit terms to the relevant Pauli operator, if they are a projector.
        operators.bit_terms[:] = operators.bit_terms[:] & 0b00_11
        self.assertEqual(operators, SparseObservable.from_list([("XZY", 1.5j), ("XZY", -0.5)]))

        boundaries = SparseObservable.from_list([("IIIIZX", 0.25), ("II+-II", 1j), ("rlIIII", 0.5)])
        boundaries.boundaries[1:-1] = [1, 5]
        self.assertEqual(
            boundaries,
            SparseObservable.from_list([("IIIIIX", 0.25), ("Il+-ZI", 1j), ("rIIIII", 0.5)]),
        )

    def test_attributes_reject_bad_writes(self):
        obs = SparseObservable.from_list([("XZY", 1.5j), ("+-r", -0.5)])
        with self.assertRaises(TypeError):
            obs.coeffs[0] = [0.25j, 0.5j]
        with self.assertRaises(TypeError):
            obs.bit_terms[0] = [SparseObservable.BitTerm.PLUS] * 4
        with self.assertRaises(TypeError):
            obs.indices[0] = [0, 1]
        with self.assertRaises(TypeError):
            obs.boundaries[0] = (0, 1)
        with self.assertRaisesRegex(ValueError, "not a valid letter"):
            obs.bit_terms[0] = 0
        with self.assertRaisesRegex(ValueError, "not a valid letter"):
            obs.bit_terms[:] = 0
        with self.assertRaisesRegex(
            ValueError, "tried to set a slice of length 2 with a sequence of length 1"
        ):
            obs.coeffs[:] = [1.0j]
        with self.assertRaisesRegex(
            ValueError, "tried to set a slice of length 6 with a sequence of length 8"
        ):
            obs.bit_terms[:] = [SparseObservable.BitTerm.Z] * 8

    def test_attributes_sequence(self):
        """Test attributes of the `Sequence` protocol."""
        # Length
        obs = SparseObservable.from_list([("XZY", 1.5j), ("+-r", -0.5)])
        self.assertEqual(len(obs.coeffs), 2)
        self.assertEqual(len(obs.indices), 6)
        self.assertEqual(len(obs.bit_terms), 6)
        self.assertEqual(len(obs.boundaries), 3)

        # Iteration
        self.assertEqual(list(obs.coeffs), [1.5j, -0.5])
        self.assertEqual(tuple(obs.indices), (0, 1, 2, 0, 1, 2))
        self.assertEqual(next(iter(obs.boundaries)), 0)
        # multiple iteration through same object
        bit_terms = obs.bit_terms
        self.assertEqual(set(bit_terms), {SparseObservable.BitTerm[x] for x in "XYZ+-r"})
        self.assertEqual(set(bit_terms), {SparseObservable.BitTerm[x] for x in "XYZ+-r"})

        # Implicit iteration methods.
        self.assertIn(SparseObservable.BitTerm.PLUS, obs.bit_terms)
        self.assertNotIn(4, obs.indices)
        self.assertEqual(list(reversed(obs.coeffs)), [-0.5, 1.5j])

        # Index by scalar
        self.assertEqual(obs.coeffs[1], -0.5)
        self.assertEqual(obs.indices[-1], 2)
        self.assertEqual(obs.bit_terms[0], SparseObservable.BitTerm.Y)
        # Make sure that Rust-space actually returns the enum value, not just an `int` (which could
        # have compared equal).
        self.assertIsInstance(obs.bit_terms[0], SparseObservable.BitTerm)
        self.assertEqual(obs.boundaries[-2], 3)
        with self.assertRaises(IndexError):
            _ = obs.coeffs[10]
        with self.assertRaises(IndexError):
            _ = obs.boundaries[-4]

        # Index by slice.  This is API guaranteed to be a Numpy array to make it easier to
        # manipulate subslices with mathematic operations.
        self.assertIsInstance(obs.coeffs[:], np.ndarray)
        np.testing.assert_array_equal(
            obs.coeffs[:], np.array([1.5j, -0.5], dtype=np.complex128), strict=True
        )
        self.assertIsInstance(obs.indices[::-1], np.ndarray)
        np.testing.assert_array_equal(
            obs.indices[::-1], np.array([2, 1, 0, 2, 1, 0], dtype=np.uint32), strict=True
        )
        self.assertIsInstance(obs.bit_terms[2:4], np.ndarray)
        np.testing.assert_array_equal(
            obs.bit_terms[2:4],
            np.array([SparseObservable.BitTerm.X, SparseObservable.BitTerm.RIGHT], dtype=np.uint8),
            strict=True,
        )
        self.assertIsInstance(obs.boundaries[-2:-3:-1], np.ndarray)
        np.testing.assert_array_equal(
            obs.boundaries[-2:-3:-1], np.array([3], dtype=np.uintp), strict=True
        )

    def test_attributes_to_array(self):
        obs = SparseObservable.from_list([("XZY", 1.5j), ("+-r", -0.5)])

        # Natural dtypes.
        np.testing.assert_array_equal(
            obs.coeffs, np.array([1.5j, -0.5], dtype=np.complex128), strict=True
        )
        np.testing.assert_array_equal(
            obs.indices, np.array([0, 1, 2, 0, 1, 2], dtype=np.uint32), strict=True
        )
        np.testing.assert_array_equal(
            obs.bit_terms,
            np.array([SparseObservable.BitTerm[x] for x in "YZXr-+"], dtype=np.uint8),
            strict=True,
        )
        np.testing.assert_array_equal(
            obs.boundaries, np.array([0, 3, 6], dtype=np.uintp), strict=True
        )

        # Cast dtypes.
        np.testing.assert_array_equal(
            np.array(obs.indices, dtype=np.uint8),
            np.array([0, 1, 2, 0, 1, 2], dtype=np.uint8),
            strict=True,
        )
        np.testing.assert_array_equal(
            np.array(obs.boundaries, dtype=np.int64),
            np.array([0, 3, 6], dtype=np.int64),
            strict=True,
        )

    @unittest.skipIf(
        int(np.__version__.split(".", maxsplit=1)[0]) < 2,
        "Numpy 1.x did not have a 'copy' keyword parameter to 'numpy.asarray'",
    )
    def test_attributes_reject_no_copy_array(self):
        obs = SparseObservable.from_list([("XZY", 1.5j), ("+-r", -0.5)])
        with self.assertRaisesRegex(ValueError, "cannot produce a safe view"):
            np.asarray(obs.coeffs, copy=False)
        with self.assertRaisesRegex(ValueError, "cannot produce a safe view"):
            np.asarray(obs.indices, copy=False)
        with self.assertRaisesRegex(ValueError, "cannot produce a safe view"):
            np.asarray(obs.bit_terms, copy=False)
        with self.assertRaisesRegex(ValueError, "cannot produce a safe view"):
            np.asarray(obs.boundaries, copy=False)

    def test_attributes_repr(self):
        # We're not testing much about the outputs here, just that they don't crash.
        obs = SparseObservable.from_list([("XZY", 1.5j), ("+-r", -0.5)])
        self.assertIn("coeffs", repr(obs.coeffs))
        self.assertIn("bit_terms", repr(obs.bit_terms))
        self.assertIn("indices", repr(obs.indices))
        self.assertIn("boundaries", repr(obs.boundaries))
