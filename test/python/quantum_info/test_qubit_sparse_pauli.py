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