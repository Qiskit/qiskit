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

from qiskit.circuit._utils import _compute_control_matrix
from qiskit.test import QiskitTestCase
from qiskit.circuit.annotated_operation import (
    AnnotatedOperation,
    ControlModifier,
    InverseModifier,
    PowerModifier,
    _canonicalize_modifiers,
)
from qiskit.circuit.library import SGate, SdgGate
from qiskit.quantum_info import Operator


class TestAnnotatedOperationlass(QiskitTestCase):
    """Testing qiskit.circuit.AnnotatedOperation"""

    def test_create_gate_with_modifier(self):
        """Test creating a gate with a single modifier."""
        op = AnnotatedOperation(SGate(), InverseModifier())
        self.assertIsInstance(op, AnnotatedOperation)
        self.assertIsInstance(op.base_op, SGate)

    def test_create_gate_with_modifier_list(self):
        """Test creating a gate with a list of modifiers."""
        op = AnnotatedOperation(
            SGate(), [InverseModifier(), ControlModifier(2), PowerModifier(3), InverseModifier()]
        )
        self.assertIsInstance(op, AnnotatedOperation)
        self.assertIsInstance(op.base_op, SGate)
        self.assertEqual(
            op.modifiers,
            [InverseModifier(), ControlModifier(2), PowerModifier(3), InverseModifier()],
        )
        self.assertNotEqual(
            op.modifiers,
            [InverseModifier(), PowerModifier(3), ControlModifier(2), InverseModifier()],
        )

    def test_create_gate_with_empty_modifier_list(self):
        """Test creating a gate with an empty list of modifiers."""
        op = AnnotatedOperation(SGate(), [])
        self.assertIsInstance(op, AnnotatedOperation)
        self.assertIsInstance(op.base_op, SGate)
        self.assertEqual(op.modifiers, [])

    def test_create_nested_annotated_gates(self):
        """Test creating an annotated gate whose base operation is also an annotated gate."""
        op_inner = AnnotatedOperation(SGate(), ControlModifier(3))
        op = AnnotatedOperation(op_inner, InverseModifier())
        self.assertIsInstance(op, AnnotatedOperation)
        self.assertIsInstance(op.base_op, AnnotatedOperation)
        self.assertIsInstance(op.base_op.base_op, SGate)

    def test_equality(self):
        """Test equality/non-equality of annotated operations
        (note that the lists of modifiers are ordered).
        """
        op1 = AnnotatedOperation(SGate(), [InverseModifier(), ControlModifier(2)])
        op2 = AnnotatedOperation(SGate(), [InverseModifier(), ControlModifier(2)])
        self.assertEqual(op1, op2)
        op3 = AnnotatedOperation(SGate(), [ControlModifier(2), InverseModifier()])
        self.assertNotEqual(op1, op3)
        op4 = AnnotatedOperation(SGate(), [InverseModifier(), ControlModifier(2, ctrl_state=2)])
        op5 = AnnotatedOperation(SGate(), [InverseModifier(), ControlModifier(2, ctrl_state=3)])
        op6 = AnnotatedOperation(SGate(), [InverseModifier(), ControlModifier(2, ctrl_state=None)])
        self.assertNotEqual(op1, op4)
        self.assertEqual(op1, op5)
        self.assertEqual(op1, op6)
        op7 = AnnotatedOperation(SdgGate(), [InverseModifier(), ControlModifier(2)])
        self.assertNotEqual(op1, op7)

    def test_num_qubits(self):
        """Tests that number of qubits is computed correctly."""
        op_inner = AnnotatedOperation(
            SGate(),
            [
                ControlModifier(4, ctrl_state=1),
                InverseModifier(),
                ControlModifier(2),
                PowerModifier(3),
                InverseModifier(),
            ],
        )
        op = AnnotatedOperation(op_inner, ControlModifier(3))
        self.assertEqual(op.num_qubits, 10)

    def test_num_clbits(self):
        """Tests that number of clbits is computed correctly."""
        op_inner = AnnotatedOperation(
            SGate(),
            [
                ControlModifier(4, ctrl_state=1),
                InverseModifier(),
                ControlModifier(2),
                PowerModifier(3),
                InverseModifier(),
            ],
        )
        op = AnnotatedOperation(op_inner, ControlModifier(3))
        self.assertEqual(op.num_clbits, 0)

    def test_to_matrix_with_control_modifier(self):
        """Test that ``to_matrix`` works correctly for control modifiers."""
        num_ctrl_qubits = 3
        for ctrl_state in [5, None, 0, 7, "110"]:
            op = AnnotatedOperation(
                SGate(), ControlModifier(num_ctrl_qubits=num_ctrl_qubits, ctrl_state=ctrl_state)
            )
            target_mat = _compute_control_matrix(SGate().to_matrix(), num_ctrl_qubits, ctrl_state)
            self.assertEqual(Operator(op), Operator(target_mat))

    def test_to_matrix_with_inverse_modifier(self):
        """Test that ``to_matrix`` works correctly for inverse modifiers."""
        op = AnnotatedOperation(SGate(), InverseModifier())
        self.assertEqual(Operator(op), Operator(SGate()).power(-1))

    def test_to_matrix_with_power_modifier(self):
        """Test that ``to_matrix`` works correctly for power modifiers with integer powers."""
        for power in [0, 1, -1, 2, -2]:
            op = AnnotatedOperation(SGate(), PowerModifier(power))
            self.assertEqual(Operator(op), Operator(SGate()).power(power))

    def test_canonicalize_modifiers(self):
        """Test that ``canonicalize_modifiers`` works correctly."""
        original_list = [
            InverseModifier(),
            ControlModifier(2),
            PowerModifier(2),
            ControlModifier(1),
            InverseModifier(),
            PowerModifier(-3),
        ]
        canonical_list = _canonicalize_modifiers(original_list)
        expected_list = [InverseModifier(), PowerModifier(6), ControlModifier(3)]
        self.assertEqual(canonical_list, expected_list)

    def test_canonicalize_inverse(self):
        """Tests that canonicalization cancels pairs of inverse modifiers."""
        original_list = _canonicalize_modifiers([InverseModifier(), InverseModifier()])
        canonical_list = _canonicalize_modifiers(original_list)
        expected_list = []
        self.assertEqual(canonical_list, expected_list)


if __name__ == "__main__":
    unittest.main()
