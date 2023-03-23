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

"""Test Qiskit's AnnotatedOperation class."""

import unittest

import numpy as np

from qiskit.circuit._utils import _compute_control_matrix
from qiskit.test import QiskitTestCase
from qiskit.circuit import QuantumCircuit, Barrier, Measure, Reset, Gate, Operation
from qiskit.circuit.annotated_operation import AnnotatedOperation, ControlModifier, InverseModifier, _canonicalize_modifiers
from qiskit.circuit.library import XGate, CXGate, SGate
from qiskit.quantum_info.operators import Clifford, CNOTDihedral, Pauli
from qiskit.extensions.quantum_initializer import Initialize, Isometry
from qiskit.quantum_info import Operator


class TestAnnotatedOperationlass(QiskitTestCase):
    """Testing qiskit.circuit.AnnotatedOperation"""

    def test_lazy_inverse(self):
        """Test that lazy inverse results in AnnotatedOperation."""
        gate = SGate()
        lazy_gate = gate.lazy_inverse()
        self.assertIsInstance(lazy_gate, AnnotatedOperation)
        self.assertIsInstance(lazy_gate.base_op, SGate)

    def test_lazy_control(self):
        """Test that lazy control results in AnnotatedOperation."""
        gate = CXGate()
        lazy_gate = gate.lazy_control(2)
        self.assertIsInstance(lazy_gate, AnnotatedOperation)
        self.assertIsInstance(lazy_gate.base_op, CXGate)

    def test_lazy_iterative(self):
        """Test that iteratively applying lazy inverse and control
        combines lazy modifiers."""
        lazy_gate = CXGate().lazy_inverse().lazy_control(2).lazy_inverse().lazy_control(1)
        self.assertIsInstance(lazy_gate, AnnotatedOperation)
        self.assertIsInstance(lazy_gate.base_op, CXGate)
        self.assertEqual(len(lazy_gate.modifiers), 4)

    def test_eq(self):
        lazy1 = CXGate().lazy_inverse().lazy_control(2)

        lazy2 = CXGate().lazy_inverse().lazy_control(2, ctrl_state=None)
        self.assertEqual(lazy1, lazy2)

        lazy3 = CXGate().lazy_inverse().lazy_control(2, ctrl_state=2)
        self.assertNotEqual(lazy1, lazy3)

        lazy4 = CXGate().lazy_inverse().lazy_control(2, ctrl_state=3)
        self.assertEqual(lazy1, lazy4)

        lazy5 = CXGate().lazy_control(2).lazy_inverse()
        self.assertNotEqual(lazy1, lazy5)

    def test_lazy_open_control(self):
        base_gate = XGate()
        base_mat = base_gate.to_matrix()
        num_ctrl_qubits = 3

        for ctrl_state in [5, None, 0, 7, "110"]:
            lazy_gate = AnnotatedOperation(base_gate, ControlModifier(num_ctrl_qubits=num_ctrl_qubits, ctrl_state=ctrl_state))
            target_mat = _compute_control_matrix(base_mat, num_ctrl_qubits, ctrl_state)
            self.assertEqual(Operator(lazy_gate), Operator(target_mat))

    def test_canonize(self):
        modifiers = [ControlModifier(num_ctrl_qubits=2, ctrl_state=None)]
        canonical_modifiers = _canonicalize_modifiers(modifiers)
        self.assertEqual(modifiers, canonical_modifiers)


if __name__ == "__main__":
    unittest.main()
