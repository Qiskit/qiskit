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
    def test_default_constructor_pauli(self):
        data = Pauli("IXYIZ")
        self.assertEqual(PauliLindbladMap(data), PauliLindbladMap.from_pauli(data))
        self.assertEqual(
            PauliLindbladMap(data, num_qubits=data.num_qubits), PauliLindbladMap.from_pauli(data)
        )
        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            PauliLindbladMap(data, num_qubits=data.num_qubits + 1)

        with_phase = Pauli("-IYYXY")
        self.assertEqual(PauliLindbladMap(with_phase), PauliLindbladMap.from_pauli(with_phase))
        self.assertEqual(
            PauliLindbladMap(with_phase, num_qubits=data.num_qubits),
            PauliLindbladMap.from_pauli(with_phase),
        )

        with self.assertRaisesRegex(ValueError, "real phases"):
            PauliLindbladMap(Pauli("-jIYYXY"))

        self.assertEqual(PauliLindbladMap(Pauli("")), PauliLindbladMap.from_pauli(Pauli("")))
    
    def test_default_constructor_sparse_pauli_op(self):
        data = SparsePauliOp.from_list([("IIXIY", 1.0), ("XYYZI", -0.25), ("XYIYY", -0.75)])
        self.assertEqual(PauliLindbladMap(data), PauliLindbladMap.from_sparse_pauli_op(data))
        self.assertEqual(
            PauliLindbladMap(data, num_qubits=data.num_qubits),
            PauliLindbladMap.from_sparse_pauli_op(data),
        )

        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            PauliLindbladMap(data, num_qubits=data.num_qubits + 1)
        with self.assertRaisesRegex(TypeError, "complex-typed coefficients"):
            PauliLindbladMap(SparsePauliOp(["XX"], [Parameter("x")]))
        with self.assertRaisesRegex(ValueError, "real-valued coefficients"):
            PauliLindbladMap(SparsePauliOp(["jXX"]))
        with self.assertRaisesRegex(ValueError, "real-valued coefficients"):
            PauliLindbladMap(SparsePauliOp.from_list([("IIXIY", 1.0), ("XYYZI", -0.25j)]))