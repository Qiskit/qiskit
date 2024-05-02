# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test OptimizeAnnotated pass"""

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.classical import expr, types
from qiskit.circuit.library import SwapGate, CXGate, HGate
from qiskit.circuit.annotated_operation import (
    AnnotatedOperation,
    ControlModifier,
    InverseModifier,
    PowerModifier,
)
from qiskit.transpiler.passes import OptimizeAnnotated
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestOptimizeSwapBeforeMeasure(QiskitTestCase):
    """Test optimizations related to annotated operations."""

    def test_combine_modifiers(self):
        """Test that the pass correctly combines modifiers."""
        gate1 = AnnotatedOperation(
            SwapGate(),
            [
                InverseModifier(),
                ControlModifier(2),
                PowerModifier(4),
                InverseModifier(),
                ControlModifier(1),
                PowerModifier(-0.5),
            ],
        )
        gate2 = AnnotatedOperation(SwapGate(), [InverseModifier(), InverseModifier()])
        gate3 = AnnotatedOperation(
            AnnotatedOperation(CXGate(), ControlModifier(2)), ControlModifier(1)
        )
        gate4 = AnnotatedOperation(
            AnnotatedOperation(SwapGate(), InverseModifier()), InverseModifier()
        )
        gate5 = CXGate()

        gate1_expected = AnnotatedOperation(
            SwapGate(), [InverseModifier(), PowerModifier(2), ControlModifier(3)]
        )
        gate2_expected = SwapGate()
        gate3_expected = AnnotatedOperation(CXGate(), ControlModifier(3))
        gate4_expected = SwapGate()
        gate5_expected = CXGate()

        qc = QuantumCircuit(6)
        qc.append(gate1, [3, 2, 4, 0, 5])
        qc.append(gate2, [1, 5])
        qc.append(gate3, [5, 4, 3, 2, 1])
        qc.append(gate4, [1, 2])
        qc.append(gate5, [4, 2])

        qc_optimized = OptimizeAnnotated()(qc)

        qc_expected = QuantumCircuit(6)
        qc_expected.append(gate1_expected, [3, 2, 4, 0, 5])
        qc_expected.append(gate2_expected, [1, 5])
        qc_expected.append(gate3_expected, [5, 4, 3, 2, 1])
        qc_expected.append(gate4_expected, [1, 2])
        qc_expected.append(gate5_expected, [4, 2])

        self.assertEqual(qc_optimized, qc_expected)

    def test_optimize_definitions(self):
        """Test that the pass descends into gate definitions when basis_gates are defined."""
        qc_def = QuantumCircuit(3)
        qc_def.cx(0, 2)
        qc_def.append(AnnotatedOperation(CXGate(), [InverseModifier(), InverseModifier()]), [0, 1])
        qc_def.append(
            AnnotatedOperation(
                SwapGate(), [InverseModifier(), ControlModifier(1), InverseModifier()]
            ),
            [0, 1, 2],
        )

        expected_qc_def_optimized = QuantumCircuit(3)
        expected_qc_def_optimized.cx(0, 2)
        expected_qc_def_optimized.cx(0, 1)
        expected_qc_def_optimized.append(
            AnnotatedOperation(SwapGate(), ControlModifier(1)), [0, 1, 2]
        )

        gate = Gate("custom_gate", 3, [])
        gate.definition = qc_def

        qc = QuantumCircuit(4)
        qc.h(0)
        qc.append(gate, [0, 1, 3])

        qc_optimized = OptimizeAnnotated(basis_gates=["cx", "u"])(qc)
        self.assertEqual(qc_optimized[1].operation.definition, expected_qc_def_optimized)

    def test_do_not_optimize_definitions_without_basis_gates(self):
        """
        Test that the pass does not descend into gate definitions when neither the
        target nor basis_gates are defined.
        """
        qc_def = QuantumCircuit(3)
        qc_def.cx(0, 2)
        qc_def.append(AnnotatedOperation(CXGate(), [InverseModifier(), InverseModifier()]), [0, 1])
        qc_def.append(
            AnnotatedOperation(
                SwapGate(), [InverseModifier(), ControlModifier(1), InverseModifier()]
            ),
            [0, 1, 2],
        )

        gate = Gate("custom_gate", 3, [])
        gate.definition = qc_def

        qc = QuantumCircuit(4)
        qc.h(0)
        qc.append(gate, [0, 1, 3])

        qc_optimized = OptimizeAnnotated()(qc)
        self.assertEqual(qc_optimized[1].operation.definition, qc_def)

    def test_do_not_optimize_definitions_without_recurse(self):
        """
        Test that the pass does not descend into gate definitions when recurse is
        False.
        """
        qc_def = QuantumCircuit(3)
        qc_def.cx(0, 2)
        qc_def.append(AnnotatedOperation(CXGate(), [InverseModifier(), InverseModifier()]), [0, 1])
        qc_def.append(
            AnnotatedOperation(
                SwapGate(), [InverseModifier(), ControlModifier(1), InverseModifier()]
            ),
            [0, 1, 2],
        )

        gate = Gate("custom_gate", 3, [])
        gate.definition = qc_def

        qc = QuantumCircuit(4)
        qc.h(0)
        qc.append(gate, [0, 1, 3])

        qc_optimized = OptimizeAnnotated(basis_gates=["cx", "u"], recurse=False)(qc)
        self.assertEqual(qc_optimized[1].operation.definition, qc_def)

    def test_if_else(self):
        """Test optimizations with if-else block."""

        true_body = QuantumCircuit(3)
        true_body.h(0)
        true_body.append(
            AnnotatedOperation(CXGate(), [InverseModifier(), InverseModifier()]), [0, 1]
        )
        false_body = QuantumCircuit(3)
        false_body.append(
            AnnotatedOperation(
                SwapGate(), [InverseModifier(), ControlModifier(1), InverseModifier()]
            ),
            [0, 1, 2],
        )

        qc = QuantumCircuit(3, 1)
        qc.h(0)
        qc.measure(0, 0)
        qc.if_else((0, True), true_body, false_body, [0, 1, 2], [])

        qc_optimized = OptimizeAnnotated()(qc)

        expected_true_body_optimized = QuantumCircuit(3)
        expected_true_body_optimized.h(0)
        expected_true_body_optimized.append(CXGate(), [0, 1])
        expected_false_body_optimized = QuantumCircuit(3)
        expected_false_body_optimized.append(
            AnnotatedOperation(SwapGate(), ControlModifier(1)), [0, 1, 2]
        )

        expected_qc = QuantumCircuit(3, 1)
        expected_qc.h(0)
        expected_qc.measure(0, 0)
        expected_qc.if_else(
            (0, True), expected_true_body_optimized, expected_false_body_optimized, [0, 1, 2], []
        )

        self.assertEqual(qc_optimized, expected_qc)

    def test_standalone_var(self):
        """Test that standalone vars work."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))

        qc = QuantumCircuit(3, 3, inputs=[a])
        qc.add_var(b, 12)
        qc.append(AnnotatedOperation(HGate(), [ControlModifier(1), ControlModifier(1)]), [0, 1, 2])
        qc.append(AnnotatedOperation(CXGate(), [InverseModifier(), InverseModifier()]), [0, 1])
        qc.measure([0, 1, 2], [0, 1, 2])
        qc.store(a, expr.logic_and(qc.clbits[0], qc.clbits[1]))

        expected = qc.copy_empty_like()
        expected.store(b, 12)
        expected.append(HGate().control(2, annotated=True), [0, 1, 2])
        expected.cx(0, 1)
        expected.measure([0, 1, 2], [0, 1, 2])
        expected.store(a, expr.logic_and(expected.clbits[0], expected.clbits[1]))

        self.assertEqual(OptimizeAnnotated()(qc), expected)
