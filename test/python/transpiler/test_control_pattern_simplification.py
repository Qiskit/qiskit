# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the ControlPatternSimplification pass."""

import unittest
from test import QiskitTestCase  

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZGate
from qiskit.quantum_info import Operator
from qiskit.transpiler.passes import ControlPatternSimplification


class TestControlPatternSimplification(QiskitTestCase):
    """Tests for ControlPatternSimplification."""

    def _verify_circuits_equivalent(self, qc1, qc2, msg="Circuits are not equivalent"):
        """Check that two circuits are unitarily equivalent."""
        self.assertEqual(Operator(qc1), Operator(qc2), msg)

    def test_complementary_11_01(self):
        """Complementary patterns ['11', '01'] reduce to single control on q0."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(3)
        unsimplified_qc.append(
            RXGate(theta).control(2, ctrl_state="11", annotated=False), [0, 1, 2]
        )
        unsimplified_qc.append(
            RXGate(theta).control(2, ctrl_state="01", annotated=False), [0, 1, 2]
        )

        expected_qc = QuantumCircuit(3)
        expected_qc.append(RXGate(theta).control(1, ctrl_state="1", annotated=False), [0, 2])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_complementary_10_00(self):
        """Complementary patterns ['10', '00'] reduce to inverted control on q0."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(3)
        unsimplified_qc.append(
            RXGate(theta).control(2, ctrl_state="10", annotated=False), [0, 1, 2]
        )
        unsimplified_qc.append(
            RXGate(theta).control(2, ctrl_state="00", annotated=False), [0, 1, 2]
        )

        expected_qc = QuantumCircuit(3)
        expected_qc.append(RXGate(theta).control(1, ctrl_state="0", annotated=False), [0, 2])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_complementary_11_10(self):
        """Complementary patterns ['11', '10'] reduce to single control on q1."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(3)
        unsimplified_qc.append(
            RXGate(theta).control(2, ctrl_state="11", annotated=False), [0, 1, 2]
        )
        unsimplified_qc.append(
            RXGate(theta).control(2, ctrl_state="10", annotated=False), [0, 1, 2]
        )

        expected_qc = QuantumCircuit(3)
        expected_qc.append(RXGate(theta).control(1, ctrl_state="1", annotated=False), [1, 2])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_complementary_01_00(self):
        """Complementary patterns ['01', '00'] reduce to inverted control on q1."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(3)
        unsimplified_qc.append(
            RXGate(theta).control(2, ctrl_state="01", annotated=False), [0, 1, 2]
        )
        unsimplified_qc.append(
            RXGate(theta).control(2, ctrl_state="00", annotated=False), [0, 1, 2]
        )

        expected_qc = QuantumCircuit(3)
        expected_qc.append(RXGate(theta).control(1, ctrl_state="0", annotated=False), [1, 2])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_complementary_gate_agnostic_ry(self):
        """Complementary patterns work for RY gates."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(3)
        unsimplified_qc.append(
            RYGate(theta).control(2, ctrl_state="11", annotated=False), [0, 1, 2]
        )
        unsimplified_qc.append(
            RYGate(theta).control(2, ctrl_state="01", annotated=False), [0, 1, 2]
        )

        expected_qc = QuantumCircuit(3)
        expected_qc.append(RYGate(theta).control(1, ctrl_state="1", annotated=False), [0, 2])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_complementary_gate_agnostic_rz(self):
        """Complementary patterns work for RZ gates."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(3)
        unsimplified_qc.append(
            RZGate(theta).control(2, ctrl_state="11", annotated=False), [0, 1, 2]
        )
        unsimplified_qc.append(
            RZGate(theta).control(2, ctrl_state="01", annotated=False), [0, 1, 2]
        )

        expected_qc = QuantumCircuit(3)
        expected_qc.append(RZGate(theta).control(1, ctrl_state="1", annotated=False), [0, 2])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_subset_111_110(self):
        """Patterns ['111', '110'] reduce from 3 to 2 controls."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(4)
        unsimplified_qc.append(
            RXGate(theta).control(3, ctrl_state="111", annotated=False), [0, 1, 2, 3]
        )
        unsimplified_qc.append(
            RXGate(theta).control(3, ctrl_state="110", annotated=False), [0, 1, 2, 3]
        )

        expected_qc = QuantumCircuit(4)
        expected_qc.append(RXGate(theta).control(2, ctrl_state="11", annotated=False), [1, 2, 3])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_subset_3control_111_011(self):
        """Patterns ['111', '011'] reduce from 3 to 2 controls."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(4)
        unsimplified_qc.append(
            RXGate(theta).control(3, ctrl_state="111", annotated=False), [0, 1, 2, 3]
        )
        unsimplified_qc.append(
            RXGate(theta).control(3, ctrl_state="011", annotated=False), [0, 1, 2, 3]
        )

        expected_qc = QuantumCircuit(4)
        expected_qc.append(RXGate(theta).control(2, ctrl_state="11", annotated=False), [0, 1, 3])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_subset_111_101(self):
        """Patterns ['111', '101'] reduce from 3 to 2 controls."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(4)
        unsimplified_qc.append(
            RXGate(theta).control(3, ctrl_state="111", annotated=False), [0, 1, 2, 3]
        )
        unsimplified_qc.append(
            RXGate(theta).control(3, ctrl_state="101", annotated=False), [0, 1, 2, 3]
        )

        expected_qc = QuantumCircuit(4)
        expected_qc.append(RXGate(theta).control(2, ctrl_state="11", annotated=False), [0, 2, 3])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_complete_partition_2qubits(self):
        """All 4 two-qubit patterns reduce to unconditional gate."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(3)
        unsimplified_qc.append(
            RXGate(theta).control(2, ctrl_state="00", annotated=False), [0, 1, 2]
        )
        unsimplified_qc.append(
            RXGate(theta).control(2, ctrl_state="01", annotated=False), [0, 1, 2]
        )
        unsimplified_qc.append(
            RXGate(theta).control(2, ctrl_state="10", annotated=False), [0, 1, 2]
        )
        unsimplified_qc.append(
            RXGate(theta).control(2, ctrl_state="11", annotated=False), [0, 1, 2]
        )

        expected_qc = QuantumCircuit(3)
        expected_qc.append(RXGate(theta), [2])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_complete_partition_3qubits(self):
        """All 8 three-qubit patterns reduce to unconditional gate."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(4)
        for pattern in ["000", "001", "010", "011", "100", "101", "110", "111"]:
            unsimplified_qc.append(
                RXGate(theta).control(3, ctrl_state=pattern, annotated=False), [0, 1, 2, 3]
            )

        expected_qc = QuantumCircuit(4)
        expected_qc.rx(theta, 3)

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_boolean_simplification_3patterns_drop_q1_3to2gates(self):
        """Three patterns with one complementary pair reduce to 2 gates."""
        theta = np.pi / 2

        unsimplified_qc = QuantumCircuit(5)
        unsimplified_qc.append(
            RXGate(theta).control(4, ctrl_state="0000", annotated=False), [1, 2, 3, 4, 0]
        )
        unsimplified_qc.append(
            RXGate(theta).control(4, ctrl_state="0100", annotated=False), [1, 2, 3, 4, 0]
        )
        unsimplified_qc.append(
            RXGate(theta).control(4, ctrl_state="0001", annotated=False), [1, 2, 3, 4, 0]
        )

        expected_qc = QuantumCircuit(5)
        expected_qc.append(
            RXGate(theta).control(3, ctrl_state="000", annotated=False), [2, 3, 4, 0]
        )
        expected_qc.append(
            RXGate(theta).control(4, ctrl_state="0100", annotated=False), [1, 2, 3, 4, 0]
        )

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_esop_synthesis_5patterns_boolean_and_xor_5to3rx_gates(self):
        """Five 6-qubit patterns reduce to 3 RX gates via boolean + XOR optimization."""
        theta = np.pi / 2

        unsimplified_qc = QuantumCircuit(7)
        unsimplified_qc.append(
            RXGate(theta).control(6, ctrl_state="000000", annotated=False), [1, 2, 3, 4, 5, 6, 0]
        )
        unsimplified_qc.append(
            RXGate(theta).control(6, ctrl_state="100000", annotated=False), [1, 2, 3, 4, 5, 6, 0]
        )
        unsimplified_qc.append(
            RXGate(theta).control(6, ctrl_state="001000", annotated=False), [1, 2, 3, 4, 5, 6, 0]
        )
        unsimplified_qc.append(
            RXGate(theta).control(6, ctrl_state="000100", annotated=False), [1, 2, 3, 4, 5, 6, 0]
        )
        unsimplified_qc.append(
            RXGate(theta).control(6, ctrl_state="000001", annotated=False), [1, 2, 3, 4, 5, 6, 0]
        )

        expected_qc = QuantumCircuit(7)
        expected_qc.append(
            RXGate(theta).control(5, ctrl_state="00000", annotated=False), [1, 2, 3, 4, 5, 0]
        )

        expected_qc.cx(3, 4)
        expected_qc.append(
            RXGate(theta).control(5, ctrl_state="00100", annotated=False), [1, 2, 4, 5, 6, 0]
        )
        expected_qc.cx(3, 4)
        expected_qc.append(
            RXGate(theta).control(6, ctrl_state="100000", annotated=False), [1, 2, 3, 4, 5, 6, 0]
        )

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_identical_patterns_angle_merge(self):
        """Identical patterns merge angles into a single gate."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(4)
        unsimplified_qc.append(
            RXGate(theta).control(3, ctrl_state="110", annotated=False), [0, 1, 2, 3]
        )
        unsimplified_qc.append(
            RXGate(theta).control(3, ctrl_state="110", annotated=False), [0, 1, 2, 3]
        )

        expected_qc = QuantumCircuit(4)
        expected_qc.append(
            RXGate(2 * theta).control(3, ctrl_state="110", annotated=False), [0, 1, 2, 3]
        )

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_identical_3control(self):
        """Identical 3-control patterns merge angles."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(4)
        unsimplified_qc.append(
            RXGate(theta).control(3, ctrl_state="110", annotated=False), [0, 1, 2, 3]
        )
        unsimplified_qc.append(
            RXGate(theta).control(3, ctrl_state="110", annotated=False), [0, 1, 2, 3]
        )

        expected_qc = QuantumCircuit(4)
        expected_qc.append(
            RXGate(2 * theta).control(3, ctrl_state="110", annotated=False), [0, 1, 2, 3]
        )

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_xor_standard_10_01_2to3gates(self):
        """XOR patterns ['10', '01'] optimize to 1 RX + 2 CX gates."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(3)
        unsimplified_qc.append(
            RXGate(theta).control(2, ctrl_state="10", annotated=False), [0, 1, 2]
        )
        unsimplified_qc.append(
            RXGate(theta).control(2, ctrl_state="01", annotated=False), [0, 1, 2]
        )

        expected_qc = QuantumCircuit(3)
        expected_qc.cx(0, 1)
        expected_qc.append(RXGate(theta).control(1, ctrl_state="1", annotated=False), [1, 2])
        expected_qc.cx(0, 1)

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_xor_with_common_factor_110_101_2to3gates(self):
        """XOR with common factor: ['110', '101'] optimize to 1 RX + 2 CX gates."""
        theta = np.pi / 2

        unsimplified_qc = QuantumCircuit(4)
        unsimplified_qc.append(
            RXGate(theta).control(3, ctrl_state="110", annotated=False), [0, 1, 2, 3]
        )
        unsimplified_qc.append(
            RXGate(theta).control(3, ctrl_state="101", annotated=False), [0, 1, 2, 3]
        )

        expected_qc = QuantumCircuit(4)
        expected_qc.cx(1, 2)
        expected_qc.append(RXGate(theta).control(2, ctrl_state="11", annotated=False), [0, 2, 3])
        expected_qc.cx(1, 2)

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_subset_different_control_counts(self):
        """Different control counts cannot be simplified."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(3)
        unsimplified_qc.append(
            RXGate(theta).control(2, ctrl_state="11", annotated=False), [0, 1, 2]
        )
        unsimplified_qc.append(RXGate(theta).control(1, ctrl_state="1", annotated=False), [0, 2])

        expected_qc = QuantumCircuit(3)
        expected_qc.append(RXGate(theta).control(2, ctrl_state="11", annotated=False), [0, 1, 2])
        expected_qc.append(RXGate(theta).control(1, ctrl_state="1", annotated=False), [0, 2])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)
        self.assertLessEqual(len(optimized_qc.data), 2, "Should not increase gate count")

    def test_partial_xor_3patterns_00_10_01_3to4gates(self):
        """Partial XOR: 3 patterns with one XOR pair reduce to 2 RX + 2 CX gates."""
        theta = np.pi / 2

        unsimplified_qc = QuantumCircuit(3)
        unsimplified_qc.append(
            RXGate(theta).control(2, ctrl_state="00", annotated=False), [0, 1, 2]
        )
        unsimplified_qc.append(
            RXGate(theta).control(2, ctrl_state="10", annotated=False), [0, 1, 2]
        )
        unsimplified_qc.append(
            RXGate(theta).control(2, ctrl_state="01", annotated=False), [0, 1, 2]
        )

        expected_qc = QuantumCircuit(3)
        expected_qc.append(RXGate(theta).control(2, ctrl_state="00", annotated=False), [0, 1, 2])
        expected_qc.cx(0, 1)
        expected_qc.append(RXGate(theta).control(1, ctrl_state="1", annotated=False), [1, 2])
        expected_qc.cx(0, 1)

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_no_optimization_different_parameters(self):
        """Different rotation angles prevent optimization."""
        theta1 = np.pi / 4
        theta2 = np.pi / 2

        unsimplified_qc = QuantumCircuit(3)
        unsimplified_qc.append(
            RXGate(theta1).control(2, ctrl_state="11", annotated=False), [0, 1, 2]
        )
        unsimplified_qc.append(
            RXGate(theta2).control(2, ctrl_state="01", annotated=False), [0, 1, 2]
        )

        expected_qc = QuantumCircuit(3)
        expected_qc.append(RXGate(theta1).control(2, ctrl_state="11", annotated=False), [0, 1, 2])
        expected_qc.append(RXGate(theta2).control(2, ctrl_state="01", annotated=False), [0, 1, 2])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_no_optimization_different_targets(self):
        """Different target qubits prevent optimization."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(4)
        unsimplified_qc.append(
            RXGate(theta).control(2, ctrl_state="11", annotated=False), [0, 1, 2]
        )
        unsimplified_qc.append(
            RXGate(theta).control(2, ctrl_state="01", annotated=False), [0, 1, 3]
        )

        expected_qc = QuantumCircuit(4)
        expected_qc.append(RXGate(theta).control(2, ctrl_state="11", annotated=False), [0, 1, 2])
        expected_qc.append(RXGate(theta).control(2, ctrl_state="01", annotated=False), [0, 1, 3])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_no_optimization_mixed_gate_types(self):
        """Different gate types (RX vs RY) prevent optimization."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(3)
        unsimplified_qc.append(
            RXGate(theta).control(2, ctrl_state="11", annotated=False), [0, 1, 2]
        )
        unsimplified_qc.append(
            RYGate(theta).control(2, ctrl_state="01", annotated=False), [0, 1, 2]
        )

        expected_qc = QuantumCircuit(3)
        expected_qc.append(RXGate(theta).control(2, ctrl_state="11", annotated=False), [0, 1, 2])
        expected_qc.append(RYGate(theta).control(2, ctrl_state="01", annotated=False), [0, 1, 2])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_xor_with_x_gates_11_00_pattern_2to5gates(self):
        """XOR 11-00 pattern: ['011', '000'] optimize to 1 RX + 2 X + 2 CX gates."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(4)
        unsimplified_qc.append(
            RXGate(theta).control(3, ctrl_state="011", annotated=False), [0, 2, 3, 1]
        )
        unsimplified_qc.append(
            RXGate(theta).control(3, ctrl_state="000", annotated=False), [0, 2, 3, 1]
        )

        expected_qc = QuantumCircuit(4)
        expected_qc.x(3)
        expected_qc.cx(2, 3)
        expected_qc.append(RXGate(theta).control(2, ctrl_state="01", annotated=False), [0, 3, 1])
        expected_qc.cx(2, 3)
        expected_qc.x(3)

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_xor_with_x_gates_00_11_pattern_2to5gates(self):
        """XOR 00-11 pattern: ['010', '111'] optimize to 1 RX + 2 X + 2 CX gates."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(4)
        unsimplified_qc.append(
            RXGate(theta).control(3, ctrl_state="010", annotated=False), [0, 1, 3, 2]
        )
        unsimplified_qc.append(
            RXGate(theta).control(3, ctrl_state="111", annotated=False), [0, 1, 3, 2]
        )

        expected_qc = QuantumCircuit(4)
        expected_qc.x(0)
        expected_qc.cx(0, 3)
        expected_qc.append(RXGate(theta).control(2, ctrl_state="11", annotated=False), [1, 3, 2])
        expected_qc.cx(0, 3)
        expected_qc.x(0)

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_3controlled_rx_xor_000_111_to_cx_mcrx(self):
        """XOR pattern ['000', '111'] reduces to CX sandwich + 2-controlled RX."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(4)
        unsimplified_qc.append(
            RXGate(theta).control(3, ctrl_state="000", annotated=False), [0, 1, 2, 3]
        )
        unsimplified_qc.append(
            RXGate(theta).control(3, ctrl_state="111", annotated=False), [0, 1, 2, 3]
        )

        expected_qc = QuantumCircuit(4)
        expected_qc.cx(0, 1)
        expected_qc.cx(0, 2)
        expected_qc.append(RXGate(theta).control(2, ctrl_state="00", annotated=False), [1, 2, 3])
        expected_qc.cx(0, 2)
        expected_qc.cx(0, 1)

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertLessEqual(
            len(optimized_qc.data),
            len(expected_qc.data),
            "Optimized circuit should have at most the expected gate count",
        )

    def test_complement_3of4_patterns_to_unconditional_plus_negative(self):
        """3 of 4 patterns reduce to unconditional + negative controlled gate."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(3)
        unsimplified_qc.append(
            RZGate(theta).control(2, ctrl_state="00", annotated=False), [0, 1, 2]
        )
        unsimplified_qc.append(
            RZGate(theta).control(2, ctrl_state="01", annotated=False), [0, 1, 2]
        )
        unsimplified_qc.append(
            RZGate(theta).control(2, ctrl_state="10", annotated=False), [0, 1, 2]
        )

        expected_qc = QuantumCircuit(3)
        expected_qc.rz(theta, 2)
        expected_qc.append(RZGate(-theta).control(2, ctrl_state="11", annotated=False), [0, 1, 2])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)
        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertEqual(len(optimized_qc.data), 2)

    def test_complement_7of8_patterns_3qubits(self):
        """7 of 8 three-qubit patterns reduce to unconditional + negative controlled gate."""
        theta = np.pi / 3

        unsimplified_qc = QuantumCircuit(4)
        for i in range(7):
            ctrl_state = format(i, "03b")
            unsimplified_qc.append(
                RYGate(theta).control(3, ctrl_state=ctrl_state, annotated=False), [0, 1, 2, 3]
            )

        expected_qc = QuantumCircuit(4)
        expected_qc.ry(theta, 3)
        expected_qc.append(
            RYGate(-theta).control(3, ctrl_state="111", annotated=False), [0, 1, 2, 3]
        )

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)
        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertEqual(len(optimized_qc.data), 2)

    def test_complement_example_00_10_11(self):
        """Patterns [00, 10, 11] missing 01 reduce to unconditional + negative gate."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(3)
        unsimplified_qc.append(
            RZGate(theta).control(2, ctrl_state="00", annotated=False), [0, 1, 2]
        )
        unsimplified_qc.append(
            RZGate(theta).control(2, ctrl_state="10", annotated=False), [0, 1, 2]
        )
        unsimplified_qc.append(
            RZGate(theta).control(2, ctrl_state="11", annotated=False), [0, 1, 2]
        )

        expected_qc = QuantumCircuit(3)
        expected_qc.rz(theta, 2)
        expected_qc.append(RZGate(-theta).control(2, ctrl_state="01", annotated=False), [0, 1, 2])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)
        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertEqual(len(optimized_qc.data), 2)

    def test_mixed_control_counts_to_unconditional(self):
        """Mixed control counts forming complete partition reduce to unconditional gate."""
        theta = np.pi / 4

        unsimplified_qc = QuantumCircuit(3)
        unsimplified_qc.append(RZGate(theta).control(1, ctrl_state="0", annotated=False), [0, 2])
        unsimplified_qc.append(
            RZGate(theta).control(2, ctrl_state="01", annotated=False), [0, 1, 2]
        )
        unsimplified_qc.append(
            RZGate(theta).control(2, ctrl_state="11", annotated=False), [0, 1, 2]
        )

        expected_qc = QuantumCircuit(3)
        expected_qc.rz(theta, 2)

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, expected_qc)
        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertEqual(len(optimized_qc.data), 1)

    def test_gate_ordering_preserved(self):
        """H gate on target qubit prevents merging of separated controlled gates."""
        theta = np.pi / 4

        qc = QuantumCircuit(3)
        qc.append(RXGate(theta).control(2, ctrl_state="11", annotated=False), [0, 1, 2])
        qc.h(2)
        qc.append(RXGate(theta).control(2, ctrl_state="01", annotated=False), [0, 1, 2])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(qc)

        self._verify_circuits_equivalent(qc, optimized_qc)

        self.assertGreaterEqual(len(optimized_qc.data), 3)

    def test_interleaved_non_controlled_gates(self):
        """X gate on control qubit prevents merging of separated controlled gates."""
        theta = np.pi / 4

        qc = QuantumCircuit(3)
        qc.append(RXGate(theta).control(2, ctrl_state="11", annotated=False), [0, 1, 2])
        qc.x(0)
        qc.append(RXGate(theta).control(2, ctrl_state="01", annotated=False), [0, 1, 2])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(qc)

        self._verify_circuits_equivalent(qc, optimized_qc)

        self.assertGreaterEqual(len(optimized_qc.data), 3)

    def test_swapped_control_qubit_order(self):
        """Swapped control qubit order is normalized before merging."""
        unsimplified_qc = QuantumCircuit(3)
        unsimplified_qc.append(RXGate(0.1).control(2, ctrl_state=2, annotated=False), [0, 1, 2])
        unsimplified_qc.append(RXGate(0.1).control(2, annotated=False), [1, 0, 2])
        unsimplified_qc.append(RXGate(0.1).control(2, annotated=False), [0, 1, 2])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(unsimplified_qc)

        self._verify_circuits_equivalent(unsimplified_qc, optimized_qc)

        self.assertEqual(len(optimized_qc.data), 2)

    def test_commuting_non_adjacent_gates_merged(self):
        """Non-adjacent gates separated by a commuting gate can be merged."""
        qc = QuantumCircuit(4)
        qc.append(RXGate(0.1).control(2, ctrl_state=2, annotated=False), [0, 1, 2])
        qc.append(RXGate(0.1).control(2, annotated=False), [0, 1, 3])
        qc.append(RXGate(0.1).control(2, ctrl_state=2, annotated=False), [0, 1, 2])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(qc)

        self._verify_circuits_equivalent(qc, optimized_qc)

        self.assertEqual(len(optimized_qc.data), 2)

    def test_non_controlled_gate_on_target_blocks_merge(self):
        """Y gate on target qubit between controlled gates prevents merging."""
        theta = np.pi / 4

        qc = QuantumCircuit(3)
        qc.append(RXGate(theta).control(2, ctrl_state=2, annotated=False), [0, 1, 2])
        qc.append(RXGate(theta).control(2, annotated=False), [0, 1, 2])
        qc.y(2)

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(qc)

        self._verify_circuits_equivalent(qc, optimized_qc)


if __name__ == "__main__":
    unittest.main()
