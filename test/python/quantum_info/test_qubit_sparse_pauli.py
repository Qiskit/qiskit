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
import itertools
import pickle
import random
import unittest

import ddt
import numpy as np

from qiskit import transpile
from qiskit.circuit import Measure, Parameter, library, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import QubitSparsePauli, QubitSparsePauliList, SparsePauliOp, Pauli, PauliList
from qiskit.transpiler import Target

from test import QiskitTestCase, combine  # pylint: disable=wrong-import-order


def single_cases():
    return [
        QubitSparsePauliList.empty(0),
        QubitSparsePauliList.empty(10),
        QubitSparsePauliList.from_list(["YIXZII", "ZZYYXX"]),
        # Includes a duplicate entry.
        QubitSparsePauliList.from_list(["IXZ", "ZZI", "IXZ"]),
    ]

@ddt.ddt
class TesQubitSparsePauli(QiskitTestCase):
    pass

@ddt.ddt
class TesQubitSparsePauliList(QiskitTestCase):
    def test_default_constructor_list(self):
        data = ["IXIIZ", "XIXII", "IIXYI"]
        self.assertEqual(QubitSparsePauliList(data), QubitSparsePauliList.from_list(data))
        self.assertEqual(QubitSparsePauliList(data, num_qubits=5), QubitSparsePauliList.from_list(data))
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
            QubitSparsePauliList([], num_qubits=5), QubitSparsePauliList.from_sparse_list([], num_qubits=5)
        )
    
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
        terms = np.full((num_qubits,), QubitSparsePauliList.BitTerm.Z, dtype=np.uint8)
        indices = np.arange(num_qubits, dtype=np.uint32)
        boundaries = np.arange(num_qubits + 1, dtype=np.uintp)
        qubit_sparse_pauli_list = QubitSparsePauliList.from_raw_parts(
            num_qubits, terms, indices, boundaries
        )
        self.assertEqual(qubit_sparse_pauli_list.num_qubits, num_qubits)
        np.testing.assert_equal(qubit_sparse_pauli_list.bit_terms, terms)
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
        np.testing.assert_equal(qubit_sparse_pauli_list.bit_terms, terms)
        np.testing.assert_equal(qubit_sparse_pauli_list.indices, indices)
        np.testing.assert_equal(qubit_sparse_pauli_list.boundaries, boundaries)

        # Construction of an empty list.
        self.assertEqual(
            QubitSparsePauliList.from_raw_parts(10, [], [], [0]), QubitSparsePauliList.empty(10)
        )

        # Construction of an operator with an intermediate identity term.  For the initial
        # constructor tests, it's hard to check anything more than the construction succeeded.
        self.assertEqual(
            QubitSparsePauliList.from_raw_parts(
                10, [1, 3, 2], [0, 1, 2], [0, 1, 1, 3]
            ).num_terms,
            3,
        )

    def test_from_raw_parts_checks_coherence(self):
        with self.assertRaisesRegex(ValueError, "not a valid letter"):
            QubitSparsePauliList.from_raw_parts(2, [ord("$")], [0], [0, 1])
        with self.assertRaisesRegex(ValueError, r"`bit_terms` \(1\) and `indices` \(0\)"):
            QubitSparsePauliList.from_raw_parts(2, [QubitSparsePauliList.BitTerm.Z], [], [0, 1])
        with self.assertRaisesRegex(ValueError, r"`bit_terms` \(0\) and `indices` \(1\)"):
            QubitSparsePauliList.from_raw_parts(2, [], [1], [0, 1])
        with self.assertRaisesRegex(ValueError, r"the first item of `boundaries` \(1\) must be 0"):
            QubitSparsePauliList.from_raw_parts(2, [QubitSparsePauliList.BitTerm.Z], [0], [1, 1])
        with self.assertRaisesRegex(ValueError, r"the last item of `boundaries` \(2\)"):
            QubitSparsePauliList.from_raw_parts(2, [1], [0], [0, 2])
        with self.assertRaisesRegex(ValueError, r"the last item of `boundaries` \(1\)"):
            QubitSparsePauliList.from_raw_parts(2, [1, 2], [0, 1], [0, 1])
        with self.assertRaisesRegex(ValueError, r"the last item of `boundaries` \(0\)"):
            QubitSparsePauliList.from_raw_parts(2, [QubitSparsePauliList.BitTerm.Z], [0], [0])
        with self.assertRaisesRegex(ValueError, r"all qubit indices must be less than the number"):
            QubitSparsePauliList.from_raw_parts(4, [1, 2], [0, 4], [0, 2])
        with self.assertRaisesRegex(ValueError, r"all qubit indices must be less than the number"):
            QubitSparsePauliList.from_raw_parts(4, [1, 2], [0, 4], [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "the values in `boundaries` include backwards"):
            QubitSparsePauliList.from_raw_parts(
                5, [1, 2, 3, 2], [0, 1, 2, 3], [0, 2, 1, 4]
            )
        with self.assertRaisesRegex(
            ValueError, "the values in `indices` are not term-wise increasing"
        ):
            QubitSparsePauliList.from_raw_parts(4, [1, 2], [1, 0], [0, 2])

        # There's no test of attempting to pass incoherent data and `check=False` because that
        # permits undefined behaviour in Rust (it's unsafe), so all bets would be off.

    def test_from_list(self):
        label = "IXYIZZY"
        self.assertEqual(
            QubitSparsePauliList.from_list([label]),
            QubitSparsePauliList.from_raw_parts(
                len(label),
                [
                    QubitSparsePauliList.BitTerm.Y,
                    QubitSparsePauliList.BitTerm.Z,
                    QubitSparsePauliList.BitTerm.Z,
                    QubitSparsePauliList.BitTerm.Y,
                    QubitSparsePauliList.BitTerm.X,
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
                    QubitSparsePauliList.BitTerm.Y,
                    QubitSparsePauliList.BitTerm.Z,
                    QubitSparsePauliList.BitTerm.Z,
                    QubitSparsePauliList.BitTerm.Y,
                    QubitSparsePauliList.BitTerm.X,
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
                    QubitSparsePauliList.BitTerm.Z,
                    QubitSparsePauliList.BitTerm.X,
                    QubitSparsePauliList.BitTerm.X,
                    QubitSparsePauliList.BitTerm.X,
                ],
                [1, 2, 4, 5],
                [0, 2, 4],
            ),
        )

        self.assertEqual(QubitSparsePauliList.from_list([], num_qubits=5), QubitSparsePauliList.empty(5))
        self.assertEqual(QubitSparsePauliList.from_list([], num_qubits=0), QubitSparsePauliList.empty(0))

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
            QubitSparsePauliList.from_list(["IIZ","II"])
        with self.assertRaisesRegex(ValueError, "label with length 3 cannot be added"):
            QubitSparsePauliList.from_list(["IIZ", "IXI"], num_qubits=2)
        with self.assertRaisesRegex(ValueError, "label with length 3 cannot be added"):
            QubitSparsePauliList.from_list(["IIZ","IXI"], num_qubits=4)
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
            QubitSparsePauliList.from_list(["IIIYX","IXIXI", "ZIYIY"]),
        )

        # The indices should be allowed to be given in unsorted order, but they should be term-wise
        # sorted in the output.
        from_unsorted = QubitSparsePauliList.from_sparse_list(
            [
                ("XYZ", (2, 1, 0)),
                ("XYY", (2, 0, 1)),
            ],
            num_qubits=3,
        )
        self.assertEqual(from_unsorted, QubitSparsePauliList.from_list(["XYZ", "XYY"]))
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
            QubitSparsePauliList.from_sparse_list(
                [("XZ", (1, 0)), ("YX", (1, 0))], num_qubits=10
            ),
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

    def test_from_terms(self):
        self.assertEqual(
            QubitSparsePauliList.from_terms([], num_qubits=5), QubitSparsePauliList.empty(5)
        )
        self.assertEqual(
            QubitSparsePauliList.from_terms((), num_qubits=0), QubitSparsePauliList.empty(0)
        )
        self.assertEqual(
            QubitSparsePauliList.from_terms((None for _ in []), num_qubits=3),
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
        self.assertEqual(QubitSparsePauliList.from_terms(list(expected)), expected)
        self.assertEqual(QubitSparsePauliList.from_terms(tuple(expected)), expected)
        self.assertEqual(QubitSparsePauliList.from_terms(term for term in expected), expected)
        self.assertEqual(
            QubitSparsePauliList.from_terms(
                (term for term in expected), num_qubits=expected.num_qubits
            ),
            expected,
        )

    def test_from_terms_failures(self):
        with self.assertRaisesRegex(ValueError, "cannot construct.*without knowing `num_qubits`"):
            QubitSparsePauliList.from_terms([])

        left, right = (
            QubitSparsePauliList(["IIXYI"])[0],
            QubitSparsePauliList(["IIIIIIIIX"])[0],
        )
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits"):
            QubitSparsePauliList.from_terms([left, right])
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits"):
            QubitSparsePauliList.from_terms([left], num_qubits=100)

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
        self.assertEqual(
            QubitSparsePauliList.from_list(["IIIXIZ", "YYXXII"]).num_terms, 2
        )

    def test_empty(self):
        empty_5 = QubitSparsePauliList.empty(5)
        self.assertEqual(empty_5.num_qubits, 5)
        np.testing.assert_equal(empty_5.bit_terms, np.array([], dtype=np.uint8))
        np.testing.assert_equal(empty_5.indices, np.array([], dtype=np.uint32))
        np.testing.assert_equal(empty_5.boundaries, np.array([0], dtype=np.uintp))

        empty_0 = QubitSparsePauliList.empty(0)
        self.assertEqual(empty_0.num_qubits, 0)
        np.testing.assert_equal(empty_0.bit_terms, np.array([], dtype=np.uint8))
        np.testing.assert_equal(empty_0.indices, np.array([], dtype=np.uint32))
        np.testing.assert_equal(empty_0.boundaries, np.array([0], dtype=np.uintp))

    def test_len(self):
        self.assertEqual(len(QubitSparsePauliList.empty(0)), 0)
        self.assertEqual(len(QubitSparsePauliList.empty(10)), 0)
        self.assertEqual(len(QubitSparsePauliList.from_list(["IIIXIZ", "YYXXII"])), 2)

    def test_bit_term_enum(self):
        # These are very explicit tests that effectively just duplicate magic numbers, but the point
        # is that those magic numbers are required to be constant as their values are part of the
        # public interface.

        self.assertEqual(
            set(QubitSparsePauliList.BitTerm),
            {
                QubitSparsePauliList.BitTerm.X,
                QubitSparsePauliList.BitTerm.Y,
                QubitSparsePauliList.BitTerm.Z,
            },
        )
        # All the enumeration items should also be integers.
        self.assertIsInstance(QubitSparsePauliList.BitTerm.X, int)
        values = {
            "X": 0b10,
            "Y": 0b11,
            "Z": 0b01,
        }
        self.assertEqual({name: getattr(QubitSparsePauliList.BitTerm, name) for name in values}, values)

        # The single-character label aliases can be accessed with index notation.
        labels = {
            "X": QubitSparsePauliList.BitTerm.X,
            "Y": QubitSparsePauliList.BitTerm.Y,
            "Z": QubitSparsePauliList.BitTerm.Z,
        }
        self.assertEqual({label: QubitSparsePauliList.BitTerm[label] for label in labels}, labels)
        # The `label` property returns known values.
        self.assertEqual(
            {bit_term.label: bit_term for bit_term in QubitSparsePauliList.BitTerm}, labels
        )

    @ddt.idata(single_cases())
    def test_pickle(self, qubit_sparse_pauli_list):
        self.assertEqual(qubit_sparse_pauli_list, copy.copy(qubit_sparse_pauli_list))
        self.assertIsNot(qubit_sparse_pauli_list, copy.copy(qubit_sparse_pauli_list))
        self.assertEqual(qubit_sparse_pauli_list, copy.deepcopy(qubit_sparse_pauli_list))
        self.assertEqual(qubit_sparse_pauli_list, pickle.loads(pickle.dumps(qubit_sparse_pauli_list)))