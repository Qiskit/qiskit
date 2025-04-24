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
from qiskit.quantum_info import PauliLindbladMap, SparsePauliOp, Pauli, PauliList
from qiskit.transpiler import Target

from test import QiskitTestCase, combine  # pylint: disable=wrong-import-order

@ddt.ddt
class TestPauliLindbladMap(QiskitTestCase):
    

    def test_from_raw_parts(self):
        # Happiest path: exactly typed inputs.
        num_qubits = 100
        terms = np.full((num_qubits,), PauliLindbladMap.BitTerm.Z, dtype=np.uint8)
        indices = np.arange(num_qubits, dtype=np.uint32)
        coeffs = np.ones((num_qubits,), dtype=float)
        boundaries = np.arange(num_qubits + 1, dtype=np.uintp)
        pauli_lindblad_map = PauliLindbladMap.from_raw_parts(num_qubits, coeffs, terms, indices, boundaries)
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

        # At least the initial implementation of `SparseObservable` requires `from_raw_parts` to be
        # a copy constructor in order to allow it to be resized by Rust space.  This is checking for
        # that, but if the implementation changes, it could potentially be relaxed.
        self.assertFalse(np.may_share_memory(pauli_lindblad_map.coeffs, coeffs))

        # Conversion from array-likes, including mis-typed but compatible arrays.
        pauli_lindblad_map = PauliLindbladMap.from_raw_parts(
            num_qubits, list(coeffs), tuple(terms), pauli_lindblad_map.indices, boundaries.astype(np.int16)
        )
        self.assertEqual(pauli_lindblad_map.num_qubits, num_qubits)
        np.testing.assert_equal(pauli_lindblad_map.bit_terms, terms)
        np.testing.assert_equal(pauli_lindblad_map.indices, indices)
        np.testing.assert_equal(pauli_lindblad_map.coeffs, coeffs)
        np.testing.assert_equal(pauli_lindblad_map.boundaries, boundaries)

        # Construction of zero operator.
        self.assertEqual(
            PauliLindbladMap.from_raw_parts(10, [], [], [], [0]), PauliLindbladMap.zero(10)
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
        with self.assertRaisesRegex(ValueError, r"boundaries.*must be one element longer"):
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
                5, [1.0j, -0.5, 2.0], [1, 2, 3, 2], [0, 1, 2, 3], [0, 2, 1, 4]
            )
        with self.assertRaisesRegex(
            ValueError, "the values in `indices` are not term-wise increasing"
        ):
            PauliLindbladMap.from_raw_parts(4, [1.0], [1, 2], [1, 0], [0, 2])

        # There's no test of attempting to pass incoherent data and `check=False` because that
        # permits undefined behaviour in Rust (it's unsafe), so all bets would be off.


    def test_from_label(self):
        # The label is interpreted like a bitstring, with the right-most item associated with qubit
        # 0, and increasing as we move to the left (like `Pauli`, and other bitstring conventions).
        self.assertEqual(
            # Ruler for counting terms:  dcba9876543210
            PauliLindbladMap.from_label("IXXIIZYIXYIXYZ"),
            PauliLindbladMap.from_raw_parts(
                14,
                [1.0],
                [
                    PauliLindbladMap.BitTerm.Z,
                    PauliLindbladMap.BitTerm.Y,
                    PauliLindbladMap.BitTerm.X,
                    PauliLindbladMap.BitTerm.Y,
                    PauliLindbladMap.BitTerm.X,
                    PauliLindbladMap.BitTerm.Y,
                    PauliLindbladMap.BitTerm.Z,
                    PauliLindbladMap.BitTerm.X,
                    PauliLindbladMap.BitTerm.X,
                ],
                [0, 1, 2, 4, 5, 7, 8, 11, 12],
                [0, 9],
            ),
        )
    
    def test_from_label_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            PauliLindbladMap.from_label("I+-$%I")
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Unicode shenangigans.
            PauliLindbladMap.from_label("🐍")
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # +/-
            PauliLindbladMap.from_label("I+-I")