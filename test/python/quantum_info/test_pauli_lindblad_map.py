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

from qiskit.quantum_info import QubitSparsePauliList, PauliLindbladMap

from test import QiskitTestCase  # pylint: disable=wrong-import-order


def single_cases():
    return [
        PauliLindbladMap.identity(0),
        PauliLindbladMap.identity(10),
        PauliLindbladMap.from_list([("YIXZII", -0.25), ("ZZYYXX", 0.25)]),
        # Includes a duplicate entry.
        PauliLindbladMap.from_list([("IXZ", -0.25), ("ZZI", 0.25), ("IXZ", 0.75)]),
    ]


@ddt.ddt
class TestPauliLindbladMap(QiskitTestCase):

    def test_default_constructor_list(self):
        data = [("IXIIZ", 0.5), ("XIXII", 1.0), ("IIXYI", -0.75)]
        self.assertEqual(PauliLindbladMap(data), PauliLindbladMap.from_list(data))
        self.assertEqual(PauliLindbladMap(data, num_qubits=5), PauliLindbladMap.from_list(data))
        with self.assertRaisesRegex(ValueError, "label with length 5 cannot be added"):
            PauliLindbladMap(data, num_qubits=4)
        with self.assertRaisesRegex(ValueError, "label with length 5 cannot be added"):
            PauliLindbladMap(data, num_qubits=6)
        self.assertEqual(
            PauliLindbladMap([], num_qubits=5), PauliLindbladMap.from_list([], num_qubits=5)
        )

    def test_default_constructor_sparse_list(self):
        data = [("ZX", (0, 3), 0.5), ("XY", (2, 4), 1.0), ("ZY", (2, 1), -0.75)]
        self.assertEqual(
            PauliLindbladMap(data, num_qubits=5),
            PauliLindbladMap.from_sparse_list(data, num_qubits=5),
        )
        self.assertEqual(
            PauliLindbladMap(data, num_qubits=10),
            PauliLindbladMap.from_sparse_list(data, num_qubits=10),
        )
        with self.assertRaisesRegex(ValueError, "'num_qubits' must be provided"):
            PauliLindbladMap(data)
        self.assertEqual(
            PauliLindbladMap([], num_qubits=5), PauliLindbladMap.from_sparse_list([], num_qubits=5)
        )

    def test_default_constructor_copy(self):
        base = PauliLindbladMap.from_list([("IXIZIY", 1.0), ("XYZIII", -1.0)])
        copied = PauliLindbladMap(base)
        self.assertEqual(base, copied)
        self.assertIsNot(base, copied)

        # Modifications to `copied` don't propagate back.
        copied.coeffs[1] = -0.5
        self.assertNotEqual(base, copied)

        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            PauliLindbladMap(base, num_qubits=base.num_qubits + 1)

    def test_default_constructor_term(self):
        expected = PauliLindbladMap.from_list([("IIZXII", 2)])
        self.assertEqual(PauliLindbladMap(expected[0]), expected)

    def test_default_constructor_term_iterable(self):
        expected = PauliLindbladMap.from_list([("IIZXII", 2), ("IIIIII", 0.5)])
        terms = [expected[0], expected[1]]
        self.assertEqual(PauliLindbladMap(list(terms)), expected)
        self.assertEqual(PauliLindbladMap(tuple(terms)), expected)
        self.assertEqual(PauliLindbladMap(term for term in terms), expected)

    def test_from_raw_parts(self):
        # Happiest path: exactly typed inputs.
        num_qubits = 100
        terms = np.full((num_qubits,), PauliLindbladMap.BitTerm.Z, dtype=np.uint8)
        indices = np.arange(num_qubits, dtype=np.uint32)
        coeffs = np.ones((num_qubits,), dtype=float)
        boundaries = np.arange(num_qubits + 1, dtype=np.uintp)
        pauli_lindblad_map = PauliLindbladMap.from_raw_parts(
            num_qubits, coeffs, terms, indices, boundaries
        )
        self.assertEqual(pauli_lindblad_map.num_qubits, num_qubits)
        np.testing.assert_equal(pauli_lindblad_map.bit_terms, terms)
        np.testing.assert_equal(pauli_lindblad_map.indices, indices)
        np.testing.assert_equal(pauli_lindblad_map.coeffs, coeffs)
        np.testing.assert_equal(pauli_lindblad_map.boundaries, boundaries)

        self.assertEqual(
            pauli_lindblad_map,
            PauliLindbladMap.from_raw_parts(
                num_qubits, coeffs, terms, indices, boundaries, check=False
            ),
        )

        # At least the initial implementation of `PauliLindbladMap` requires `from_raw_parts` to be
        # a copy constructor in order to allow it to be resized by Rust space.  This is checking for
        # that, but if the implementation changes, it could potentially be relaxed.
        self.assertFalse(np.may_share_memory(pauli_lindblad_map.coeffs, coeffs))

        # Conversion from array-likes, including mis-typed but compatible arrays.
        pauli_lindblad_map = PauliLindbladMap.from_raw_parts(
            num_qubits,
            list(coeffs),
            tuple(terms),
            pauli_lindblad_map.indices,
            boundaries.astype(np.int16),
        )
        self.assertEqual(pauli_lindblad_map.num_qubits, num_qubits)
        np.testing.assert_equal(pauli_lindblad_map.bit_terms, terms)
        np.testing.assert_equal(pauli_lindblad_map.indices, indices)
        np.testing.assert_equal(pauli_lindblad_map.coeffs, coeffs)
        np.testing.assert_equal(pauli_lindblad_map.boundaries, boundaries)

        # Construction of identity operator.
        self.assertEqual(
            PauliLindbladMap.from_raw_parts(10, [], [], [], [0]), PauliLindbladMap.identity(10)
        )

        # Construction of an operator with an intermediate identity term.  For the initial
        # constructor tests, it's hard to check anything more than the construction succeeded.
        self.assertEqual(
            PauliLindbladMap.from_raw_parts(
                10, [1.0, 0.5, 2.0], [1, 3, 2], [0, 1, 2], [0, 1, 1, 3]
            ).num_terms,
            # The three are [(1.0)*(Z_1), 0.5, 2.0*(X_2 Y_1)]
            3,
        )

    def test_from_raw_parts_checks_coherence(self):
        with self.assertRaisesRegex(ValueError, "not a valid letter"):
            PauliLindbladMap.from_raw_parts(2, [1.0], [ord("$")], [0], [0, 1])
        with self.assertRaisesRegex(ValueError, r"must match the length of `bit_terms`"):
            PauliLindbladMap.from_raw_parts(2, [1.0], [PauliLindbladMap.BitTerm.Z], [0], [0])
        with self.assertRaisesRegex(ValueError, r"`bit_terms` \(1\) and `indices` \(0\)"):
            PauliLindbladMap.from_raw_parts(2, [1.0], [PauliLindbladMap.BitTerm.Z], [], [0, 1])
        with self.assertRaisesRegex(ValueError, r"`bit_terms` \(0\) and `indices` \(1\)"):
            PauliLindbladMap.from_raw_parts(2, [1.0], [], [1], [0, 1])
        with self.assertRaisesRegex(ValueError, r"the first item of `boundaries` \(1\) must be 0"):
            PauliLindbladMap.from_raw_parts(2, [1.0], [PauliLindbladMap.BitTerm.Z], [0], [1, 1])
        with self.assertRaisesRegex(ValueError, r"the last item of `boundaries` \(2\)"):
            PauliLindbladMap.from_raw_parts(2, [1.0], [1], [0], [0, 2])
        with self.assertRaisesRegex(ValueError, r"the last item of `boundaries` \(1\)"):
            PauliLindbladMap.from_raw_parts(2, [1.0], [1, 2], [0, 1], [0, 1])
        with self.assertRaisesRegex(ValueError, r"all qubit indices must be less than the number"):
            PauliLindbladMap.from_raw_parts(4, [1.0], [1, 2], [0, 4], [0, 2])
        with self.assertRaisesRegex(ValueError, r"all qubit indices must be less than the number"):
            PauliLindbladMap.from_raw_parts(4, [1.0, -0.5], [1, 2], [0, 4], [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "the values in `boundaries` include backwards"):
            PauliLindbladMap.from_raw_parts(
                5, [1.0, -0.5, 2.0], [1, 2, 3, 2], [0, 1, 2, 3], [0, 2, 1, 4]
            )
        with self.assertRaisesRegex(
            ValueError, "the values in `indices` are not term-wise increasing"
        ):
            PauliLindbladMap.from_raw_parts(4, [1.0], [1, 2], [1, 0], [0, 2])

        # There's no test of attempting to pass incoherent data and `check=False` because that
        # permits undefined behaviour in Rust (it's unsafe), so all bets would be off.

    def test_from_list(self):
        label = "IXYIZZY"
        self.assertEqual(
            PauliLindbladMap.from_list([(label, 1.0)]),
            PauliLindbladMap.from_raw_parts(
                len(label),
                [1.0],
                [
                    PauliLindbladMap.BitTerm.Y,
                    PauliLindbladMap.BitTerm.Z,
                    PauliLindbladMap.BitTerm.Z,
                    PauliLindbladMap.BitTerm.Y,
                    PauliLindbladMap.BitTerm.X,
                ],
                [0, 1, 2, 4, 5],
                [0, 5],
            ),
        )
        self.assertEqual(
            PauliLindbladMap.from_list([(label, 1.0)], num_qubits=len(label)),
            PauliLindbladMap.from_raw_parts(
                len(label),
                [1.0],
                [
                    PauliLindbladMap.BitTerm.Y,
                    PauliLindbladMap.BitTerm.Z,
                    PauliLindbladMap.BitTerm.Z,
                    PauliLindbladMap.BitTerm.Y,
                    PauliLindbladMap.BitTerm.X,
                ],
                [0, 1, 2, 4, 5],
                [0, 5],
            ),
        )

        self.assertEqual(
            PauliLindbladMap.from_list([("IIIXZI", 1.0), ("XXIIII", -0.5)]),
            PauliLindbladMap.from_raw_parts(
                6,
                [1.0, -0.5],
                [
                    PauliLindbladMap.BitTerm.Z,
                    PauliLindbladMap.BitTerm.X,
                    PauliLindbladMap.BitTerm.X,
                    PauliLindbladMap.BitTerm.X,
                ],
                [1, 2, 4, 5],
                [0, 2, 4],
            ),
        )

        self.assertEqual(PauliLindbladMap.from_list([], num_qubits=5), PauliLindbladMap.identity(5))
        self.assertEqual(PauliLindbladMap.from_list([], num_qubits=0), PauliLindbladMap.identity(0))

    def test_from_list_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            PauliLindbladMap.from_list([("XZIIZY", 0.5), ("I+-$%I", 1.0)])
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Unicode shenangigans.
            PauliLindbladMap.from_list([("🐍", 0.5)])
        with self.assertRaisesRegex(ValueError, "label with length 4 cannot be added"):
            PauliLindbladMap.from_list([("IIZ", 0.5), ("IIXI", 1.0)])
        with self.assertRaisesRegex(ValueError, "label with length 2 cannot be added"):
            PauliLindbladMap.from_list([("IIZ", 0.5), ("II", 1.0)])
        with self.assertRaisesRegex(ValueError, "label with length 3 cannot be added"):
            PauliLindbladMap.from_list([("IIZ", 0.5), ("IXI", 1.0)], num_qubits=2)
        with self.assertRaisesRegex(ValueError, "label with length 3 cannot be added"):
            PauliLindbladMap.from_list([("IIZ", 0.5), ("IXI", 1.0)], num_qubits=4)
        with self.assertRaisesRegex(ValueError, "cannot construct.*without knowing `num_qubits`"):
            PauliLindbladMap.from_list([])

    def test_from_sparse_list(self):
        self.assertEqual(
            PauliLindbladMap.from_sparse_list(
                [
                    ("XY", (0, 1), 0.5),
                    ("XX", (1, 3), -0.25),
                    ("YYZ", (0, 2, 4), 1.0),
                ],
                num_qubits=5,
            ),
            PauliLindbladMap.from_list([("IIIYX", 0.5), ("IXIXI", -0.25), ("ZIYIY", 1.0)]),
        )

        # The indices should be allowed to be given in unsorted order, but they should be term-wise
        # sorted in the output.
        from_unsorted = PauliLindbladMap.from_sparse_list(
            [
                ("XYZ", (2, 1, 0), 1.5),
                ("XYY", (2, 0, 1), -0.5),
            ],
            num_qubits=3,
        )
        self.assertEqual(from_unsorted, PauliLindbladMap.from_list([("XYZ", 1.5), ("XYY", -0.5)]))
        np.testing.assert_equal(
            from_unsorted.indices, np.array([0, 1, 2, 0, 1, 2], dtype=np.uint32)
        )

        # Explicit identities should still work, just be skipped over.
        explicit_identity = PauliLindbladMap.from_sparse_list(
            [
                ("ZXI", (0, 1, 2), 1.0),
                ("XYIII", (0, 1, 2, 3, 8), -0.5),
            ],
            num_qubits=10,
        )
        self.assertEqual(
            explicit_identity,
            PauliLindbladMap.from_sparse_list(
                [("XZ", (1, 0), 1.0), ("YX", (1, 0), -0.5)], num_qubits=10
            ),
        )
        np.testing.assert_equal(explicit_identity.indices, np.array([0, 1, 0, 1], dtype=np.uint32))

        self.assertEqual(
            PauliLindbladMap.from_sparse_list([], num_qubits=1_000_000),
            PauliLindbladMap.identity(1_000_000),
        )
        self.assertEqual(
            PauliLindbladMap.from_sparse_list([], num_qubits=0),
            PauliLindbladMap.identity(0),
        )

    def test_from_sparse_list_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            PauliLindbladMap.from_sparse_list(
                [("XZZY", (5, 3, 1, 0), 0.5), ("+$", (2, 1), 1.0)], num_qubits=8
            )
        # Unicode shenangigans.  These two should fail with a `ValueError`, but the exact message
        # isn't important.  "\xff" is "ÿ", which is two bytes in UTF-8 (so has a length of 2 in
        # Rust), but has a length of 1 in Python, so try with both a length-1 and length-2 index
        # sequence, and both should still raise `ValueError`.
        with self.assertRaises(ValueError):
            PauliLindbladMap.from_sparse_list([("\xff", (1,), 0.5)], num_qubits=5)
        with self.assertRaises(ValueError):
            PauliLindbladMap.from_sparse_list([("\xff", (1, 2), 0.5)], num_qubits=5)

        with self.assertRaisesRegex(ValueError, "label with length 2 does not match indices"):
            PauliLindbladMap.from_sparse_list([("XZ", (0,), 1.0)], num_qubits=5)
        with self.assertRaisesRegex(ValueError, "label with length 2 does not match indices"):
            PauliLindbladMap.from_sparse_list([("XZ", (0, 1, 2), 1.0)], num_qubits=5)

        with self.assertRaisesRegex(ValueError, "index 3 is out of range for a 3-qubit operator"):
            PauliLindbladMap.from_sparse_list([("XZY", (0, 1, 3), 1.0)], num_qubits=3)
        with self.assertRaisesRegex(ValueError, "index 4 is out of range for a 3-qubit operator"):
            PauliLindbladMap.from_sparse_list([("XZY", (0, 1, 4), 1.0)], num_qubits=3)
        with self.assertRaisesRegex(ValueError, "index 3 is out of range for a 3-qubit operator"):
            # ... even if it's for an explicit identity.
            PauliLindbladMap.from_sparse_list([("XXI", (0, 1, 3), 1.0)], num_qubits=3)

        with self.assertRaisesRegex(ValueError, "index 3 is duplicated"):
            PauliLindbladMap.from_sparse_list([("XZ", (3, 3), 1.0)], num_qubits=5)
        with self.assertRaisesRegex(ValueError, "index 3 is duplicated"):
            PauliLindbladMap.from_sparse_list([("XYZXZ", (3, 0, 1, 2, 3), 1.0)], num_qubits=5)

    def test_from_terms(self):
        self.assertEqual(
            PauliLindbladMap.from_terms([], num_qubits=5), PauliLindbladMap.identity(5)
        )
        self.assertEqual(
            PauliLindbladMap.from_terms((), num_qubits=0), PauliLindbladMap.identity(0)
        )
        self.assertEqual(
            PauliLindbladMap.from_terms((None for _ in []), num_qubits=3),
            PauliLindbladMap.identity(3),
        )

        expected = PauliLindbladMap.from_sparse_list(
            [
                ("XYZ", (4, 2, 1), 1),
                ("XXYY", (8, 5, 3, 2), 0.5),
                ("ZZ", (5, 0), 2.0),
            ],
            num_qubits=10,
        )
        self.assertEqual(PauliLindbladMap.from_terms(list(expected)), expected)
        self.assertEqual(PauliLindbladMap.from_terms(tuple(expected)), expected)
        self.assertEqual(PauliLindbladMap.from_terms(term for term in expected), expected)
        self.assertEqual(
            PauliLindbladMap.from_terms(
                (term for term in expected), num_qubits=expected.num_qubits
            ),
            expected,
        )

    def test_from_terms_failures(self):
        with self.assertRaisesRegex(ValueError, "cannot construct.*without knowing `num_qubits`"):
            PauliLindbladMap.from_terms([])

        left, right = (
            PauliLindbladMap([("IIXYI", 1.0)])[0],
            PauliLindbladMap([("IIIIIIIIX", 1.0)])[0],
        )
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits"):
            PauliLindbladMap.from_terms([left, right])
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits"):
            PauliLindbladMap.from_terms([left], num_qubits=100)

    def test_default_constructor_failed_inference(self):
        with self.assertRaises(TypeError):
            # Mixed dense/sparse list.
            PauliLindbladMap([("IIXIZ", 1.0), ("IZ", (2, 3), -1.0)], num_qubits=5)

    def test_num_qubits(self):
        self.assertEqual(PauliLindbladMap.identity(0).num_qubits, 0)
        self.assertEqual(PauliLindbladMap.identity(10).num_qubits, 10)

    def test_num_terms(self):
        self.assertEqual(PauliLindbladMap.identity(0).num_terms, 0)
        self.assertEqual(PauliLindbladMap.identity(10).num_terms, 0)
        self.assertEqual(
            PauliLindbladMap.from_list([("IIIXIZ", 1.0), ("YYXXII", 0.5)]).num_terms, 2
        )

    def test_identity(self):
        identity_5 = PauliLindbladMap.identity(5)
        self.assertEqual(identity_5.num_qubits, 5)
        np.testing.assert_equal(identity_5.coeffs, np.array([], dtype=float))
        np.testing.assert_equal(identity_5.bit_terms, np.array([], dtype=np.uint8))
        np.testing.assert_equal(identity_5.indices, np.array([], dtype=np.uint32))
        np.testing.assert_equal(identity_5.boundaries, np.array([0], dtype=np.uintp))

        identity_0 = PauliLindbladMap.identity(0)
        self.assertEqual(identity_0.num_qubits, 0)
        np.testing.assert_equal(identity_0.coeffs, np.array([], dtype=float))
        np.testing.assert_equal(identity_0.bit_terms, np.array([], dtype=np.uint8))
        np.testing.assert_equal(identity_0.indices, np.array([], dtype=np.uint32))
        np.testing.assert_equal(identity_0.boundaries, np.array([0], dtype=np.uintp))

    def test_len(self):
        self.assertEqual(len(PauliLindbladMap.identity(0)), 0)
        self.assertEqual(len(PauliLindbladMap.identity(10)), 0)
        self.assertEqual(len(PauliLindbladMap.from_list([("IIIXIZ", 1.0), ("YYXXII", 0.5)])), 2)

    def test_bit_term_enum(self):
        # These are very explicit tests that effectively just duplicate magic numbers, but the point
        # is that those magic numbers are required to be constant as their values are part of the
        # public interface.

        self.assertEqual(
            set(PauliLindbladMap.BitTerm),
            {
                PauliLindbladMap.BitTerm.X,
                PauliLindbladMap.BitTerm.Y,
                PauliLindbladMap.BitTerm.Z,
            },
        )
        # All the enumeration items should also be integers.
        self.assertIsInstance(PauliLindbladMap.BitTerm.X, int)
        values = {
            "X": 0b10,
            "Y": 0b11,
            "Z": 0b01,
        }
        self.assertEqual({name: getattr(PauliLindbladMap.BitTerm, name) for name in values}, values)

        # The single-character label aliases can be accessed with index notation.
        labels = {
            "X": PauliLindbladMap.BitTerm.X,
            "Y": PauliLindbladMap.BitTerm.Y,
            "Z": PauliLindbladMap.BitTerm.Z,
        }
        self.assertEqual({label: PauliLindbladMap.BitTerm[label] for label in labels}, labels)
        # The `label` property returns known values.
        self.assertEqual(
            {bit_term.label: bit_term for bit_term in PauliLindbladMap.BitTerm}, labels
        )

    @ddt.idata(single_cases())
    def test_pickle(self, pauli_lindblad_map):
        self.assertEqual(pauli_lindblad_map, copy.copy(pauli_lindblad_map))
        self.assertIsNot(pauli_lindblad_map, copy.copy(pauli_lindblad_map))
        self.assertEqual(pauli_lindblad_map, copy.deepcopy(pauli_lindblad_map))
        self.assertEqual(pauli_lindblad_map, pickle.loads(pickle.dumps(pauli_lindblad_map)))

    @ddt.data(
        # This is every combination of (0, 1, many) for (terms, qubits, non-identites per term).
        PauliLindbladMap.identity(0),
        PauliLindbladMap.identity(1),
        PauliLindbladMap.identity(10),
        PauliLindbladMap.from_list([("YIXZII", -0.25)]),
        PauliLindbladMap.from_list([("YIXZII", -0.25), ("ZZYYXX", 0.25)]),
    )
    def test_repr(self, data):
        # The purpose of this is just to test that the `repr` doesn't crash, rather than asserting
        # that it has any particular form.
        self.assertIsInstance(repr(data), str)
        self.assertIn("PauliLindbladMap", repr(data))

    @ddt.idata(single_cases())
    def test_copy(self, pauli_lindblad_map):
        self.assertEqual(pauli_lindblad_map, pauli_lindblad_map.copy())
        self.assertIsNot(pauli_lindblad_map, pauli_lindblad_map.copy())

    def test_equality(self):
        sparse_data = [("XZ", (1, 0), 0.5), ("XYY", (3, 1, 0), -0.25)]
        op = PauliLindbladMap.from_sparse_list(sparse_data, num_qubits=5)
        self.assertEqual(op, op.copy())
        # Take care that Rust space allows multiple views onto the same object.
        self.assertEqual(op, op)

        # Comparison to some other object shouldn't fail.
        self.assertNotEqual(op, None)

        # No costly automatic simplification (mathematically, these operators _are_ the same).
        self.assertNotEqual(
            PauliLindbladMap.from_list([("X", 2.0), ("X", -1.0)]), PauliLindbladMap([("X", 1.0)])
        )

        # Difference in qubit count.
        self.assertNotEqual(
            op, PauliLindbladMap.from_sparse_list(sparse_data, num_qubits=op.num_qubits + 1)
        )
        self.assertNotEqual(PauliLindbladMap.identity(2), PauliLindbladMap.identity(3))

        # Difference in coeffs.
        self.assertNotEqual(
            PauliLindbladMap.from_list([("IIXZI", 1.0), ("XXYYZ", -0.5)]),
            PauliLindbladMap.from_list([("IIXZI", 1.0), ("XXYYZ", 0.5)]),
        )
        self.assertNotEqual(
            PauliLindbladMap.from_list([("IIXZI", 1.0), ("XXYYZ", -0.5)]),
            PauliLindbladMap.from_list([("IIXZI", -1.0), ("XXYYZ", -0.5)]),
        )

        # Difference in bit terms.
        self.assertNotEqual(
            PauliLindbladMap.from_list([("IIXZI", 1.0), ("XXYYZ", -0.5)]),
            PauliLindbladMap.from_list([("IIYZI", 1.0), ("XXYYZ", -0.5)]),
        )
        self.assertNotEqual(
            PauliLindbladMap.from_list([("IIXZI", 1.0), ("XXYYZ", -0.5)]),
            PauliLindbladMap.from_list([("IIXZI", 1.0), ("XXYYY", -0.5)]),
        )

        # Difference in indices.
        self.assertNotEqual(
            PauliLindbladMap.from_list([("IIXZI", 1.0), ("XXYYZ", -0.5)]),
            PauliLindbladMap.from_list([("IXIZI", 1.0), ("XXYYZ", -0.5)]),
        )
        self.assertNotEqual(
            PauliLindbladMap.from_list([("IIXZI", 1.0), ("XIYYZ", -0.5)]),
            PauliLindbladMap.from_list([("IIXZI", 1.0), ("IXYYZ", -0.5)]),
        )

        # Difference in boundaries.
        self.assertNotEqual(
            PauliLindbladMap.from_sparse_list(
                [("XZ", (0, 1), 1.5), ("XX", (2, 3), -0.5)], num_qubits=5
            ),
            PauliLindbladMap.from_sparse_list(
                [("XZX", (0, 1, 2), 1.5), ("X", (3,), -0.5)], num_qubits=5
            ),
        )

    def test_write_into_attributes_scalar(self):
        coeffs = PauliLindbladMap.from_sparse_list(
            [("XZ", (1, 0), 1.5), ("XX", (3, 2), -1.5)], num_qubits=8
        )
        coeffs.coeffs[0] = -2.0
        self.assertEqual(
            coeffs,
            PauliLindbladMap.from_sparse_list(
                [("XZ", (1, 0), -2.0), ("XX", (3, 2), -1.5)], num_qubits=8
            ),
        )
        coeffs.coeffs[1] = 1.5
        self.assertEqual(
            coeffs,
            PauliLindbladMap.from_sparse_list(
                [("XZ", (1, 0), -2.0), ("XX", (3, 2), 1.5)], num_qubits=8
            ),
        )

        bit_terms = PauliLindbladMap.from_sparse_list(
            [("XZ", (0, 1), 1.5), ("XX", (2, 3), -1.5)], num_qubits=8
        )
        bit_terms.bit_terms[0] = PauliLindbladMap.BitTerm.Y
        bit_terms.bit_terms[3] = PauliLindbladMap.BitTerm.Z
        self.assertEqual(
            bit_terms,
            PauliLindbladMap.from_sparse_list(
                [("YZ", (0, 1), 1.5), ("XZ", (2, 3), -1.5)], num_qubits=8
            ),
        )

        indices = PauliLindbladMap.from_sparse_list(
            [("XZ", (0, 1), 1.5), ("XX", (2, 3), -1.5)], num_qubits=8
        )
        # These two sets keep the generator in term-wise increasing order.  We don't test what
        # happens if somebody violates the Rust-space requirement to be term-wise increasing.
        indices.indices[1] = 4
        indices.indices[3] = 7
        self.assertEqual(
            indices,
            PauliLindbladMap.from_sparse_list(
                [("XZ", (0, 4), 1.5), ("XX", (2, 7), -1.5)], num_qubits=8
            ),
        )

        boundaries = PauliLindbladMap.from_sparse_list(
            [("XZ", (0, 1), 1.5), ("XX", (2, 3), -1.5)], num_qubits=8
        )
        # Move a single-qubit term from the second summand into the first (the particular indices
        # ensure we remain term-wise sorted).
        boundaries.boundaries[1] += 1
        self.assertEqual(
            boundaries,
            PauliLindbladMap.from_sparse_list(
                [("XZX", (0, 1, 2), 1.5), ("X", (3,), -1.5)], num_qubits=8
            ),
        )

    def test_write_into_attributes_broadcast(self):
        coeffs = PauliLindbladMap.from_list([("XIIZI", 1.5), ("IIIYZ", -0.25), ("ZIIIY", 0.5)])
        coeffs.coeffs[:] = 1.5
        np.testing.assert_array_equal(coeffs.coeffs, [1.5, 1.5, 1.5])
        coeffs.coeffs[1:] = 1.0
        np.testing.assert_array_equal(coeffs.coeffs, [1.5, 1.0, 1.0])
        coeffs.coeffs[:2] = -0.5
        np.testing.assert_array_equal(coeffs.coeffs, [-0.5, -0.5, 1.0])
        coeffs.coeffs[::2] = 1.5
        np.testing.assert_array_equal(coeffs.coeffs, [1.5, -0.5, 1.5])
        coeffs.coeffs[::-1] = -0.5
        np.testing.assert_array_equal(coeffs.coeffs, [-0.5, -0.5, -0.5])

        # It's hard to broadcast into `indices` without breaking data coherence; the broadcasting is
        # more meant for fast modifications to `coeffs` and `bit_terms`.
        indices = PauliLindbladMap.from_list([("XIIZI", 1.5), ("IIYIZ", -0.25), ("ZIIIY", 0.5)])
        indices.indices[::2] = 1
        self.assertEqual(
            indices, PauliLindbladMap.from_list([("XIIZI", 1.5), ("IIYZI", -0.25), ("ZIIYI", 0.5)])
        )

        bit_terms = PauliLindbladMap.from_list([("XIIZI", 1.5), ("IIYIZ", -0.25), ("ZIIIY", 0.5)])
        bit_terms.bit_terms[::2] = PauliLindbladMap.BitTerm.Z
        self.assertEqual(
            bit_terms,
            PauliLindbladMap.from_list([("XIIZI", 1.5), ("IIYIZ", -0.25), ("ZIIIZ", 0.5)]),
        )
        bit_terms.bit_terms[3:1:-1] = PauliLindbladMap.BitTerm.X
        self.assertEqual(
            bit_terms,
            PauliLindbladMap.from_list([("XIIZI", 1.5), ("IIXIX", -0.25), ("ZIIIZ", 0.5)]),
        )
        bit_terms.bit_terms[bit_terms.boundaries[2] : bit_terms.boundaries[3]] = (
            PauliLindbladMap.BitTerm.X
        )
        self.assertEqual(
            bit_terms,
            PauliLindbladMap.from_list([("XIIZI", 1.5), ("IIXIX", -0.25), ("XIIIX", 0.5)]),
        )

        boundaries = PauliLindbladMap.from_list([("IIIIZX", 1), ("IIXXII", -0.5), ("YYIIII", 0.5)])
        boundaries.boundaries[1:3] = 1
        self.assertEqual(
            boundaries,
            PauliLindbladMap.from_list([("IIIIIX", 1), ("IIIIII", -0.5), ("YYXXZI", 0.5)]),
        )

    def test_write_into_attributes_slice(self):
        coeffs = PauliLindbladMap.from_list([("XIIZI", 1.5), ("IIIYZ", -0.25), ("ZIIIY", 0.5)])
        coeffs.coeffs[:] = [2.0, 0.5, -0.25]
        self.assertEqual(
            coeffs, PauliLindbladMap.from_list([("XIIZI", 2.0), ("IIIYZ", 0.5), ("ZIIIY", -0.25)])
        )
        # This should assign the coefficients in reverse order - we more usually spell it
        # `coeffs[:] = coeffs{::-1]`, but the idea is to check the set-item slicing order.
        coeffs.coeffs[::-1] = coeffs.coeffs[:]
        self.assertEqual(
            coeffs, PauliLindbladMap.from_list([("XIIZI", -0.25), ("IIIYZ", 0.5), ("ZIIIY", 2.0)])
        )

        indices = PauliLindbladMap.from_list([("IIIIZX", 0.25), ("IIXYII", 1), ("YZIIII", 0.5)])
        indices.indices[:4] = [4, 5, 1, 2]
        self.assertEqual(
            indices, PauliLindbladMap.from_list([("ZXIIII", 0.25), ("IIIXYI", 1), ("YZIIII", 0.5)])
        )

        bit_terms = PauliLindbladMap.from_list([("IIIIZX", 0.25), ("IIXXII", 1), ("YYIIII", 0.5)])
        bit_terms.bit_terms[::2] = [
            PauliLindbladMap.BitTerm.Y,
            PauliLindbladMap.BitTerm.Y,
            PauliLindbladMap.BitTerm.Z,
        ]
        self.assertEqual(
            bit_terms,
            PauliLindbladMap.from_list([("IIIIZY", 0.25), ("IIXYII", 1), ("YZIIII", 0.5)]),
        )

        boundaries = PauliLindbladMap.from_list([("IIIIZX", 0.25), ("IIXXII", 1), ("YYIIII", 0.5)])
        boundaries.boundaries[1:-1] = [1, 5]
        self.assertEqual(
            boundaries,
            PauliLindbladMap.from_list([("IIIIIX", 0.25), ("IYXXZI", 1), ("YIIIII", 0.5)]),
        )

    def test_attributes_reject_bad_writes(self):
        pauli_lindblad_map = PauliLindbladMap.from_list([("XZY", 1.5), ("XXY", -0.5)])
        with self.assertRaises(TypeError):
            pauli_lindblad_map.coeffs[0] = [0.25j, 0.5j]
        with self.assertRaises(TypeError):
            pauli_lindblad_map.coeffs[0] = 0.25j
        with self.assertRaises(TypeError):
            pauli_lindblad_map.bit_terms[0] = [PauliLindbladMap.BitTerm.X] * 4
        with self.assertRaises(TypeError):
            pauli_lindblad_map.indices[0] = [0, 1]
        with self.assertRaises(TypeError):
            pauli_lindblad_map.boundaries[0] = (0, 1)
        with self.assertRaisesRegex(ValueError, "not a valid letter"):
            pauli_lindblad_map.bit_terms[0] = 0
        with self.assertRaisesRegex(ValueError, "not a valid letter"):
            pauli_lindblad_map.bit_terms[:] = 0
        with self.assertRaisesRegex(
            ValueError, "tried to set a slice of length 2 with a sequence of length 1"
        ):
            pauli_lindblad_map.coeffs[:] = [1.0]
        with self.assertRaisesRegex(
            ValueError, "tried to set a slice of length 6 with a sequence of length 8"
        ):
            pauli_lindblad_map.bit_terms[:] = [PauliLindbladMap.BitTerm.Z] * 8

    def test_attributes_sequence(self):
        """Test attributes of the `Sequence` protocol."""
        # Length
        pauli_lindblad_map = PauliLindbladMap.from_list([("XZY", 1.5), ("ZYX", -0.5)])
        self.assertEqual(len(pauli_lindblad_map.coeffs), 2)
        self.assertEqual(len(pauli_lindblad_map.indices), 6)
        self.assertEqual(len(pauli_lindblad_map.bit_terms), 6)
        self.assertEqual(len(pauli_lindblad_map.boundaries), 3)

        # Iteration
        self.assertEqual(list(pauli_lindblad_map.coeffs), [1.5, -0.5])
        self.assertEqual(tuple(pauli_lindblad_map.indices), (0, 1, 2, 0, 1, 2))
        self.assertEqual(next(iter(pauli_lindblad_map.boundaries)), 0)
        # multiple iteration through same object
        bit_terms = pauli_lindblad_map.bit_terms
        self.assertEqual(set(bit_terms), {PauliLindbladMap.BitTerm[x] for x in "XYZZYX"})
        self.assertEqual(set(bit_terms), {PauliLindbladMap.BitTerm[x] for x in "XYZZYX"})

        # Implicit iteration methods.
        self.assertIn(PauliLindbladMap.BitTerm.Y, pauli_lindblad_map.bit_terms)
        self.assertNotIn(4, pauli_lindblad_map.indices)
        self.assertEqual(list(reversed(pauli_lindblad_map.coeffs)), [-0.5, 1.5])

        # Index by scalar
        self.assertEqual(pauli_lindblad_map.coeffs[1], -0.5)
        self.assertEqual(pauli_lindblad_map.indices[-1], 2)
        self.assertEqual(pauli_lindblad_map.bit_terms[0], PauliLindbladMap.BitTerm.Y)
        # Make sure that Rust-space actually returns the enum value, not just an `int` (which could
        # have compared equal).
        self.assertIsInstance(pauli_lindblad_map.bit_terms[0], QubitSparsePauliList.BitTerm)
        self.assertEqual(pauli_lindblad_map.boundaries[-2], 3)
        with self.assertRaises(IndexError):
            _ = pauli_lindblad_map.coeffs[10]
        with self.assertRaises(IndexError):
            _ = pauli_lindblad_map.boundaries[-4]

        # Index by slice.  This is API guaranteed to be a Numpy array to make it easier to
        # manipulate subslices with mathematic operations.
        self.assertIsInstance(pauli_lindblad_map.coeffs[:], np.ndarray)
        np.testing.assert_array_equal(
            pauli_lindblad_map.coeffs[:], np.array([1.5, -0.5], dtype=np.float64), strict=True
        )
        self.assertIsInstance(pauli_lindblad_map.indices[::-1], np.ndarray)
        np.testing.assert_array_equal(
            pauli_lindblad_map.indices[::-1],
            np.array([2, 1, 0, 2, 1, 0], dtype=np.uint32),
            strict=True,
        )
        self.assertIsInstance(pauli_lindblad_map.bit_terms[2:4], np.ndarray)
        np.testing.assert_array_equal(
            pauli_lindblad_map.bit_terms[2:4],
            np.array([PauliLindbladMap.BitTerm.X, PauliLindbladMap.BitTerm.X], dtype=np.uint8),
            strict=True,
        )
        self.assertIsInstance(pauli_lindblad_map.boundaries[-2:-3:-1], np.ndarray)
        np.testing.assert_array_equal(
            pauli_lindblad_map.boundaries[-2:-3:-1], np.array([3], dtype=np.uintp), strict=True
        )

    def test_attributes_to_array(self):
        pauli_lindblad_map = PauliLindbladMap.from_list([("XZY", 1.5), ("XYZ", -0.5)])

        # Natural dtypes.
        np.testing.assert_array_equal(
            pauli_lindblad_map.coeffs, np.array([1.5, -0.5], dtype=np.float64), strict=True
        )
        np.testing.assert_array_equal(
            pauli_lindblad_map.indices, np.array([0, 1, 2, 0, 1, 2], dtype=np.uint32), strict=True
        )
        np.testing.assert_array_equal(
            pauli_lindblad_map.bit_terms,
            np.array([PauliLindbladMap.BitTerm[x] for x in "YZXZYX"], dtype=np.uint8),
            strict=True,
        )
        np.testing.assert_array_equal(
            pauli_lindblad_map.boundaries, np.array([0, 3, 6], dtype=np.uintp), strict=True
        )

        # Cast dtypes.
        np.testing.assert_array_equal(
            np.array(pauli_lindblad_map.indices, dtype=np.uint8),
            np.array([0, 1, 2, 0, 1, 2], dtype=np.uint8),
            strict=True,
        )
        np.testing.assert_array_equal(
            np.array(pauli_lindblad_map.boundaries, dtype=np.int64),
            np.array([0, 3, 6], dtype=np.int64),
            strict=True,
        )

    @unittest.skipIf(
        int(np.__version__.split(".", maxsplit=1)[0]) < 2,
        "Numpy 1.x did not have a 'copy' keyword parameter to 'numpy.asarray'",
    )
    def test_attributes_reject_no_copy_array(self):
        pauli_lindblad_map = PauliLindbladMap.from_list([("XZY", 1.5), ("YXZ", -0.5)])
        with self.assertRaisesRegex(ValueError, "cannot produce a safe view"):
            np.asarray(pauli_lindblad_map.coeffs, copy=False)
        with self.assertRaisesRegex(ValueError, "cannot produce a safe view"):
            np.asarray(pauli_lindblad_map.indices, copy=False)
        with self.assertRaisesRegex(ValueError, "cannot produce a safe view"):
            np.asarray(pauli_lindblad_map.bit_terms, copy=False)
        with self.assertRaisesRegex(ValueError, "cannot produce a safe view"):
            np.asarray(pauli_lindblad_map.boundaries, copy=False)

    def test_attributes_repr(self):
        # We're not testing much about the outputs here, just that they don't crash.
        pauli_lindblad_map = PauliLindbladMap.from_list([("XZY", 1.5), ("YXZ", -0.5)])
        self.assertIn("coeffs", repr(pauli_lindblad_map.coeffs))
        self.assertIn("bit_terms", repr(pauli_lindblad_map.bit_terms))
        self.assertIn("indices", repr(pauli_lindblad_map.indices))
        self.assertIn("boundaries", repr(pauli_lindblad_map.boundaries))

    @ddt.idata(single_cases())
    def test_clear(self, pauli_lindblad_map):
        num_qubits = pauli_lindblad_map.num_qubits
        pauli_lindblad_map.clear()
        self.assertEqual(pauli_lindblad_map, PauliLindbladMap.identity(num_qubits))

    def test_iteration(self):
        self.assertEqual(list(PauliLindbladMap.identity(5)), [])
        self.assertEqual(tuple(PauliLindbladMap.identity(0)), ())

        pauli_lindblad_map = PauliLindbladMap.from_sparse_list(
            [
                ("XYY", (4, 2, 1), 2),
                ("", (), 0.5),
                ("ZZ", (3, 0), -0.25),
                ("XX", (2, 1), 1.0),
                ("YZ", (4, 1), 1),
            ],
            num_qubits=5,
        )
        bit_term = PauliLindbladMap.BitTerm
        expected = [
            PauliLindbladMap.Term(5, 2, [bit_term.Y, bit_term.Y, bit_term.X], [1, 2, 4]),
            PauliLindbladMap.Term(5, 0.5, [], []),
            PauliLindbladMap.Term(5, -0.25, [bit_term.Z, bit_term.Z], [0, 3]),
            PauliLindbladMap.Term(5, 1.0, [bit_term.X, bit_term.X], [1, 2]),
            PauliLindbladMap.Term(5, 1, [bit_term.Z, bit_term.Y], [1, 4]),
        ]
        self.assertEqual(list(pauli_lindblad_map), expected)

    def test_indexing(self):
        pauli_lindblad_map = PauliLindbladMap.from_sparse_list(
            [
                ("XYY", (4, 2, 1), 2),
                ("", (), 0.5),
                ("ZZ", (3, 0), -0.25),
                ("XX", (2, 1), 1.0),
                ("YZ", (4, 1), 1),
            ],
            num_qubits=5,
        )
        bit_term = PauliLindbladMap.BitTerm
        expected = [
            PauliLindbladMap.Term(5, 2, [bit_term.Y, bit_term.Y, bit_term.X], [1, 2, 4]),
            PauliLindbladMap.Term(5, 0.5, [], []),
            PauliLindbladMap.Term(5, -0.25, [bit_term.Z, bit_term.Z], [0, 3]),
            PauliLindbladMap.Term(5, 1.0, [bit_term.X, bit_term.X], [1, 2]),
            PauliLindbladMap.Term(5, 1, [bit_term.Z, bit_term.Y], [1, 4]),
        ]
        self.assertEqual(pauli_lindblad_map[0], expected[0])
        self.assertEqual(pauli_lindblad_map[-2], expected[-2])
        self.assertEqual(pauli_lindblad_map[2:4], PauliLindbladMap(expected[2:4]))
        self.assertEqual(pauli_lindblad_map[1::2], PauliLindbladMap(expected[1::2]))
        self.assertEqual(pauli_lindblad_map[:], PauliLindbladMap(expected))
        self.assertEqual(pauli_lindblad_map[-1:-4:-1], PauliLindbladMap(expected[-1:-4:-1]))

    @ddt.data(
        PauliLindbladMap.from_sparse_list([("YXZ", [2, 3, 5], -0.25)], num_qubits=6),
        PauliLindbladMap.from_list([("YIXZII", -0.25)]),
    )
    def test_term_repr(self, pauli_lindblad_map):
        # The purpose of this is just to test that the `repr` doesn't crash, rather than asserting
        # that it has any particular form.
        term = pauli_lindblad_map[0]
        self.assertIsInstance(repr(term), str)
        self.assertIn("PauliLindbladMap.Term", repr(term))

    @ddt.data(
        PauliLindbladMap.from_sparse_list([("YXZ", [2, 3, 5], -0.25)], num_qubits=6),
        PauliLindbladMap.from_list([("YIXZII", -0.25)]),
    )
    def test_term_to_pauli_lindblad_map(self, pauli_lindblad_map):
        self.assertEqual(pauli_lindblad_map[0].to_pauli_lindblad_map(), pauli_lindblad_map)
        self.assertIsNot(pauli_lindblad_map[0].to_pauli_lindblad_map(), pauli_lindblad_map)

    def test_term_equality(self):
        self.assertEqual(
            PauliLindbladMap.Term(5, 1.0, [], []), PauliLindbladMap.Term(5, 1.0, [], [])
        )
        self.assertNotEqual(
            PauliLindbladMap.Term(5, 1.0, [], []), PauliLindbladMap.Term(8, 1.0, [], [])
        )
        self.assertNotEqual(
            PauliLindbladMap.Term(5, 1.0, [], []), PauliLindbladMap.Term(5, 2.0, [], [])
        )
        self.assertNotEqual(
            PauliLindbladMap.Term(5, 1.0, [], []), PauliLindbladMap.Term(8, -1, [], [])
        )

        pauli_lindblad_map = PauliLindbladMap.from_list(
            [
                ("IIXIZ", 2),
                ("IIZIX", 2),
                ("XXIII", -1.5),
                ("XYIII", -1.5),
                ("IYIYI", 0.5),
                ("IIYIY", 0.5),
            ]
        )
        self.assertEqual(pauli_lindblad_map[0], pauli_lindblad_map[0])
        self.assertEqual(pauli_lindblad_map[1], pauli_lindblad_map[1])
        self.assertNotEqual(pauli_lindblad_map[0], pauli_lindblad_map[1])
        self.assertEqual(pauli_lindblad_map[2], pauli_lindblad_map[2])
        self.assertEqual(pauli_lindblad_map[3], pauli_lindblad_map[3])
        self.assertNotEqual(pauli_lindblad_map[2], pauli_lindblad_map[3])
        self.assertEqual(pauli_lindblad_map[4], pauli_lindblad_map[4])
        self.assertEqual(pauli_lindblad_map[5], pauli_lindblad_map[5])
        self.assertNotEqual(pauli_lindblad_map[4], pauli_lindblad_map[5])

    @ddt.data(
        PauliLindbladMap.from_sparse_list([("YXZ", [2, 3, 5], -0.25)], num_qubits=6),
        PauliLindbladMap.from_list([("YIXZII", -0.25)]),
    )
    def test_term_pickle(self, pauli_lindblad_map):
        term = pauli_lindblad_map[0]
        self.assertEqual(pickle.loads(pickle.dumps(term)), term)
        self.assertEqual(copy.copy(term), term)
        self.assertEqual(copy.deepcopy(term), term)

    def test_term_attributes(self):
        term = PauliLindbladMap([("IIXIIXZ", 5.0)])[0]
        self.assertEqual(term.num_qubits, 7)
        self.assertEqual(term.coeff, 5.0)
        np.testing.assert_equal(
            term.bit_terms,
            np.array(
                [
                    PauliLindbladMap.BitTerm.Z,
                    PauliLindbladMap.BitTerm.X,
                    PauliLindbladMap.BitTerm.X,
                ],
                dtype=np.uint8,
            ),
        )
        np.testing.assert_equal(term.indices, np.array([0, 1, 4], dtype=np.uintp))

        term = PauliLindbladMap.from_list([("IIXYZ", 0.5)])[0]
        self.assertEqual(term.num_qubits, 5)
        self.assertEqual(term.coeff, 0.5)
        self.assertEqual(
            list(term.bit_terms),
            [
                PauliLindbladMap.BitTerm.Z,
                PauliLindbladMap.BitTerm.Y,
                PauliLindbladMap.BitTerm.X,
            ],
        )
        self.assertEqual(list(term.indices), [0, 1, 2])

    def test_term_new(self):
        expected = PauliLindbladMap([("IIIXXZIII", 1.0)])[0]

        self.assertEqual(
            PauliLindbladMap.Term(
                9,
                1.0,
                [
                    PauliLindbladMap.BitTerm.Z,
                    PauliLindbladMap.BitTerm.X,
                    PauliLindbladMap.BitTerm.X,
                ],
                [3, 4, 5],
            ),
            expected,
        )

        # Constructor should allow being given unsorted inputs, and but them in the right order.
        self.assertEqual(
            PauliLindbladMap.Term(
                9,
                1.0,
                [
                    PauliLindbladMap.BitTerm.X,
                    PauliLindbladMap.BitTerm.X,
                    PauliLindbladMap.BitTerm.Z,
                ],
                [4, 5, 3],
            ),
            expected,
        )
        self.assertEqual(list(expected.indices), [3, 4, 5])

        with self.assertRaisesRegex(ValueError, "not term-wise increasing"):
            PauliLindbladMap.Term(2, 2, [PauliLindbladMap.BitTerm.X] * 2, [0, 0])

    def test_to_sparse_list(self):
        """Test converting to a sparse list."""
        with self.subTest(msg="identity"):
            pauli_lindblad_map = PauliLindbladMap.identity(100)
            expected = []
            self.assertEqual(expected, pauli_lindblad_map.to_sparse_list())

        with self.subTest(msg="IXYZ"):
            pauli_lindblad_map = PauliLindbladMap([("IXYZ", 1.0)])
            expected = [("ZYX", [0, 1, 2], 1)]
            self.assertEqual(
                canonicalize_sparse_list(expected),
                canonicalize_sparse_list(pauli_lindblad_map.to_sparse_list()),
            )

        with self.subTest(msg="multiple"):
            pauli_lindblad_map = PauliLindbladMap.from_list([("XXIZ", 0.5), ("YYIZ", -1)])
            expected = [("XXZ", [3, 2, 0], 0.5), ("ZYY", [0, 2, 3], -1)]
            self.assertEqual(
                canonicalize_sparse_list(expected),
                canonicalize_sparse_list(pauli_lindblad_map.to_sparse_list()),
            )

    def test_sparse_term_bit_labels(self):
        """Test getting the bit labels of a SparseTerm."""

        pauli_lindblad_map = PauliLindbladMap([("IXYZXYZXYZ", 1.0)])
        term = pauli_lindblad_map[0]
        indices = term.indices
        labels = term.bit_labels()

        label_dict = dict(zip(indices, labels))
        expected = dict(enumerate("ZYXZYXZYX"))

        for i, label in expected.items():
            self.assertEqual(label, label_dict[i])

        reconstructed = PauliLindbladMap.from_sparse_list(
            [(labels, indices, 1)], pauli_lindblad_map.num_qubits
        )
        self.assertEqual(pauli_lindblad_map, reconstructed)


def canonicalize_term(pauli, indices, coeff):
    # canonicalize a sparse list term by sorting by indices (which is unique as
    # indices cannot be repeated)
    idcs = np.argsort(indices)
    sorted_paulis = "".join(pauli[i] for i in idcs)
    return (sorted_paulis, np.asarray(indices)[idcs].tolist(), float(coeff))


def canonicalize_sparse_list(sparse_list):
    # sort a sparse list representation by canonicalizing the terms and then applying
    # Python's built-in sort
    canonicalized_terms = [canonicalize_term(*term) for term in sparse_list]
    return sorted(canonicalized_terms)
