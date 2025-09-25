# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Commutative Optimization pass."""

import unittest

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.circuit.library import U1Gate, RZGate, PhaseGate, UnitaryGate, PauliEvolutionGate
from qiskit.circuit.parameter import Parameter
from qiskit.transpiler.passes import CommutativeOptimization
from qiskit.quantum_info import Operator, SparsePauliOp

from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestCommutativeOptimization(QiskitTestCase):
    """Test CommutativeOptimization pass."""

    def test_merge_rx_rotations(self):
        """Test that various RX-rotations are merged."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.sxdg(0)
        qc.x(0)
        qc.cx(1, 0)
        qc.rx(np.pi / 2, 0)
        qc.sx(0)

        qct = CommutativeOptimization()(qc)

        expected = QuantumCircuit(2, global_phase=np.pi / 2)
        expected.h(0)
        expected.cx(1, 0)
        expected.rx(3 * np.pi / 2, 0)

        self.assertEqual(Operator(expected), Operator(qc))
        self.assertEqual(qct, expected)

    def test_merge_rz_rotations(self):
        """Test that various RZ-rotations get merged."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.t(0)
        qc.sdg(0)
        qc.z(0)
        qc.cz(1, 0)
        qc.tdg(0)
        qc.rz(np.pi / 2, 0)
        qc.s(0)

        qct = CommutativeOptimization()(qc)

        expected = QuantumCircuit(2, global_phase=np.pi / 2)
        expected.h(0)
        expected.cz(1, 0)
        expected.rz(3 * np.pi / 2, 0)

        self.assertEqual(Operator(expected), Operator(qc))
        self.assertEqual(qct, expected)

    def test_nested_cancellations(self):
        """
        Test that nested cancellations are performed at once.
        """
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.s(0)
        qc.x(0)
        qc.y(0)
        qc.y(0)
        qc.x(0)
        qc.sdg(0)
        qc.h(0)

        qct = CommutativeOptimization()(qc)

        # The innermost pair of y-gates cancels out; then the pair of x-gates cancels out;
        # then s and sdg cancel out; finally the outermost pair of h-gates cancels out.
        # In contrast, the CommutativeCancellation pass would only cancel the innermost
        # pair of gates.
        expected = QuantumCircuit(2)

        self.assertEqual(Operator(expected), Operator(qc))
        self.assertEqual(qct, expected)

    def test_consecutive_cancellations(self):
        """Test that consecutive cancellations are performed at once."""

        qr = QuantumRegister(2, "q")
        qc = QuantumCircuit(qr)
        qc.h(qr[0])
        qc.h(qr[0])
        qc.x(qr[0])
        qc.x(qr[0])
        qc.y(qr[0])
        qc.y(qr[0])
        qc.rz(0.5, qr[0])
        qc.rz(0.5, qr[0])
        qc.append(U1Gate(0.5), [qr[0]])
        qc.append(U1Gate(0.5), [qr[0]])
        qc.rx(0.5, qr[0])
        qc.rx(0.5, qr[0])
        qc.cx(qr[0], qr[1])
        qc.cx(qr[0], qr[1])
        qc.cy(qr[0], qr[1])
        qc.cy(qr[0], qr[1])
        qc.cz(qr[0], qr[1])
        qc.cz(qr[0], qr[1])

        qct = CommutativeOptimization()(qc)

        expected = QuantumCircuit(qr)
        expected.append(RZGate(2.0), [qr[0]])
        expected.rx(1.0, qr[0])
        expected.global_phase = 0.5

        self.assertEqual(Operator(expected), Operator(qc))
        self.assertEqual(qct, expected)

    def test_symmetric_gates_cancel(self):
        """Test that various symmetric gates cancel."""
        qc = QuantumCircuit(2)
        qc.cz(0, 1)
        qc.cz(1, 0)
        qc.swap(0, 1)
        qc.swap(1, 0)
        qc.rxx(0.1, 0, 1)
        qc.rxx(-0.1, 1, 0)
        qc.ryy(-0.3, 0, 1)
        qc.ryy(0.3, 1, 0)

        qct = CommutativeOptimization()(qc)

        expected = QuantumCircuit(2)

        self.assertEqual(Operator(expected), Operator(qc))
        self.assertEqual(qct, expected)

    def test_parametric_gates_are_merged(self):
        """Test that parametric gates can merged."""
        alpha = Parameter("alpha")
        beta = Parameter("beta")

        qc = QuantumCircuit(2)
        qc.rz(alpha, 0)
        qc.rz(0.1, 0)
        qc.p(beta, 0)

        qct = CommutativeOptimization()(qc)

        expected = QuantumCircuit(2, global_phase=0.5 * beta)
        expected.rz(alpha + beta + 0.1, 0)

        self.assertEqual(qct, expected)

    def test_unitary_gates_cancel_upto_phase(self):
        """
        Test that a pair of up-to-phase inverse unitary gates cancels,
        and the global phase of the circuit is updated correctly.
        """

        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc1.cx(0, 1)
        u1 = UnitaryGate(Operator(qc1).data)

        qc2 = QuantumCircuit(2)
        qc2.cx(0, 1)
        qc2.h(0)
        qc2.global_phase = np.pi / 3
        u2 = UnitaryGate(Operator(qc2).data)

        qc = QuantumCircuit(2)
        qc.append(u1, [0, 1])
        qc.append(u2, [0, 1])

        qct = CommutativeOptimization()(qc)

        expected = QuantumCircuit(2, global_phase=np.pi / 3)

        self.assertEqual(Operator(expected), Operator(qc))
        self.assertEqual(qct, expected)

    def test_merge_pauli_evolution_gates(self):
        """Test that the pass merges PauliEvolutionGates when appropriate."""
        op = SparsePauliOp.from_list([("IZZ", 1), ("ZII", 2), ("ZIZ", 3)])

        qc = QuantumCircuit(4)
        qc.append(PauliEvolutionGate(op, 0.7), [0, 1, 2])
        qc.append(PauliEvolutionGate(op, -0.5), [0, 1, 2])

        qct = CommutativeOptimization()(qc)

        expected = QuantumCircuit(4)
        expected.append(PauliEvolutionGate(op, 0.2), [0, 1, 2])

        self.assertEqual(Operator(expected), Operator(qc))
        self.assertEqual(qct, expected)

    def test_cancel_pauli_evolution_gates(self):
        """Test that the pass cancels PauliEvolutionGates when appropriate."""
        op = SparsePauliOp.from_list([("IZZ", 1), ("ZII", 2), ("ZIZ", 3)])

        qc = QuantumCircuit(4)
        qc.append(PauliEvolutionGate(op, 0.7), [0, 1, 2])
        qc.append(PauliEvolutionGate(op, -0.7), [0, 1, 2])

        qct = CommutativeOptimization()(qc)

        expected = QuantumCircuit(4)

        self.assertEqual(Operator(expected), Operator(qc))
        self.assertEqual(qct, expected)

    def test_2pi_multiples(self):
        """Test 2pi multiples are handled with the correct phase they introduce."""
        for eps in [0, 1e-10, -1e-10]:
            for sign in [-1, 1]:
                qc = QuantumCircuit(1)
                qc.rz(sign * np.pi + eps, 0)
                qc.rz(sign * np.pi, 0)

                with self.subTest(msg="single 2pi", sign=sign, eps=eps):
                    tqc = CommutativeOptimization()(qc)
                    self.assertEqual(0, len(tqc.count_ops()))
                    self.assertAlmostEqual(np.pi, tqc.global_phase)

            for sign_x in [-1, 1]:
                for sign_z in [-1, 1]:
                    qc = QuantumCircuit(2)
                    qc.rx(sign_x * np.pi + eps, 0)
                    qc.rx(sign_x * np.pi, 0)
                    qc.rz(sign_z * np.pi, 1)
                    qc.rz(sign_z * np.pi, 1)

                    with self.subTest(msg="two 2pi", sign_x=sign_x, sign_z=sign_z, eps=eps):
                        tqc = CommutativeOptimization()(qc)
                        self.assertEqual(0, len(tqc.count_ops()))
                        self.assertAlmostEqual(0, tqc.global_phase)

    def test_4pi_multiples(self):
        """Test 4pi multiples are removed w/o changing the global phase."""
        for eps in [0, 1e-10, -1e-10]:
            for sign in [-1, 1]:
                qc = QuantumCircuit(1)
                qc.rz(sign * np.pi + eps, 0)
                qc.rz(sign * 6 * np.pi, 0)
                qc.rz(sign * np.pi, 0)

                with self.subTest(sign=sign, eps=eps):
                    tqc = CommutativeOptimization()(qc)
                    self.assertEqual(0, len(tqc.count_ops()))
                    self.assertAlmostEqual(0, tqc.global_phase)

    def test_fixed_rotation_accumulation(self):
        """Test accumulating gates with fixed angles (T, S) works correctly."""

        # test for U1, P and RZ as target gate
        for gate_cls in [RZGate, PhaseGate, U1Gate]:
            qc = QuantumCircuit(1)
            gate = gate_cls(0.2)
            qc.append(gate, [0])
            qc.t(0)
            qc.s(0)

            tqc = CommutativeOptimization()(qc)
            self.assertTrue(np.allclose(Operator(qc).data, Operator(tqc).data))

    def test_commutative_circuit1(self):
        """A simple circuit where three CNOTs commute, the first and the last cancel."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.h(2)
        qc.cx(2, 1)
        qc.cx(0, 1)

        qct = CommutativeOptimization()(qc)

        expected = QuantumCircuit(3)
        expected.h(2)
        expected.cx(2, 1)

        self.assertEqual(Operator(expected), Operator(qc))
        self.assertEqual(qct, expected)

    def test_consecutive_cnots(self):
        """A simple circuit with two consecutive CNOTs."""

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(0, 1)

        qct = CommutativeOptimization()(qc)

        expected = QuantumCircuit(2)

        self.assertEqual(Operator(expected), Operator(qc))
        self.assertEqual(qct, expected)

    def test_alternating_cnots(self):
        """A simple circuit with two alternating CNOTs."""

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(1, 0)

        qct = CommutativeOptimization()(qc)

        self.assertEqual(qc, qct)

    def test_nested_cnots_and_rotations(self):
        """An outer pair of rotations gates and an inner pair of CNOTs."""
        qc = QuantumCircuit(2)
        qc.rx(np.pi, 0)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.rx(np.pi, 0)

        qct = CommutativeOptimization()(qc)

        # The pass merges both inner and outer pairs in one go.
        # RX(2pi) = -I = exp(i pi) I
        expected = QuantumCircuit(2, global_phase=np.pi)

        self.assertEqual(Operator(expected), Operator(qc))
        self.assertEqual(qct, expected)

    def test_cnots_across_x_on_control(self):
        """Test that CNOTs separated by X on control qubit do not cancel."""

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.x(0)
        qc.cx(0, 1)

        qct = CommutativeOptimization()(qc)

        self.assertEqual(qc, qct)

    def test_cnots_across_z_on_control(self):
        """Test that CNOTs separated by Z on control qubit cancel."""

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.z(0)
        qc.cx(0, 1)

        qct = CommutativeOptimization()(qc)

        expected = QuantumCircuit(2)
        expected.z(0)

        self.assertEqual(Operator(expected), Operator(qc))
        self.assertEqual(qct, expected)

    def test_cnots_across_t_on_control(self):
        """Test that CNOTs separated by T on control qubit cancel."""

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.t(0)
        qc.cx(0, 1)

        qct = CommutativeOptimization()(qc)

        expected = QuantumCircuit(2)
        expected.t(0)

        self.assertEqual(Operator(expected), Operator(qc))
        self.assertEqual(qct, expected)

    def test_cnots_across_rz_on_control(self):
        """Test that CNOTs separated by RZ on control qubit cancel."""

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.rz(np.pi / 3, 0)
        qc.cx(0, 1)

        qct = CommutativeOptimization()(qc)

        expected = QuantumCircuit(2)
        expected.rz(np.pi / 3, 0)

        self.assertEqual(Operator(expected), Operator(qc))
        self.assertEqual(qct, expected)

    def test_cnots_across_z_on_target(self):
        """Test that CNOTs separated by Z on target qubit do not cancel."""

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.z(1)
        qc.cx(0, 1)

        qct = CommutativeOptimization()(qc)

        self.assertEqual(qc, qct)

    def test_cnots_across_t_on_target(self):
        """Test that CNOTs separated by T on target qubit do not cancel."""

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.t(1)
        qc.cx(0, 1)

        qct = CommutativeOptimization()(qc)

        self.assertEqual(qc, qct)

    def test_cnots_across_rz_on_target(self):
        """Test that CNOTs separated by RZ on target qubit do not cancel."""

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.rz(np.pi / 3, 1)
        qc.cx(0, 1)

        qct = CommutativeOptimization()(qc)

        self.assertEqual(qc, qct)

    def test_commutative_circuit2(self):
        """
        A more complex circuit where three CNOTs commute, the first and the last
        cancel, also two X gates cancel, and two Rz gates combine.

        qr0:----.---------------.--------     qr0:-------------
                |               |
        qr1:---(+)---(+)--[X]--(+)--[X]--  =  qr1:--------(+)--
                      |                                    |
        qr2:---[Rz]---.---[Rz]-[T]--[S]--     qr2:--[U1]---.---
        """

        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.rz(np.pi / 3, 2)
        qc.cx(2, 1)
        qc.rz(np.pi / 3, 2)
        qc.t(2)
        qc.s(2)
        qc.x(1)
        qc.cx(0, 1)
        qc.x(1)

        qct = CommutativeOptimization()(qc)

        expected = QuantumCircuit(3)
        expected.cx(2, 1)
        expected.append(RZGate(np.pi * 17 / 12), [2])
        expected.global_phase = (np.pi * 17 / 12 - (2 * np.pi / 3)) / 2

        self.assertEqual(Operator(expected), Operator(qc))
        self.assertEqual(qct, expected)

    def test_commutative_circuit3(self):
        """
        A more complex circuit where three CNOTs commute, the first and the last
        cancel, also two X gates cancel, and two Rz gates combine.

        qr0:-------.------------------.-------------     qr0:-------------
                   |                  |
        qr1:------(+)------(+)--[X]--(+)-------[X]--  =  qr1:--------(+)--
                            |                                         |
        qr2:------[Rz]--.---.----.---[Rz]-[T]--[S]--     qr2:--[U1]---.---
                        |        |
        qr3:-[Rz]--[X]-(+)------(+)--[X]-[Rz]-------     qr3:--[Rz]-------
        """

        qr = QuantumRegister(4, "qr")
        qc = QuantumCircuit(qr)

        qc.cx(qr[0], qr[1])
        qc.rz(np.pi / 3, qr[2])
        qc.rz(np.pi / 3, qr[3])
        qc.x(qr[3])
        qc.cx(qr[2], qr[3])
        qc.cx(qr[2], qr[1])
        qc.cx(qr[2], qr[3])
        qc.rz(np.pi / 3, qr[2])
        qc.t(qr[2])
        qc.x(qr[3])
        qc.rz(np.pi / 3, qr[3])
        qc.s(qr[2])
        qc.x(qr[1])
        qc.cx(qr[0], qr[1])
        qc.x(qr[1])

        qct = CommutativeOptimization()(qc)

        expected = QuantumCircuit(qr)
        expected.append(RZGate(np.pi * 2 / 3), [qr[3]])
        expected.cx(qr[2], qr[1])
        expected.append(RZGate(np.pi * 17 / 12), [qr[2]])
        expected.global_phase = 3 * np.pi / 8

        self.assertEqual(Operator(expected), Operator(qc))
        self.assertEqual(qct, expected)

    def test_cnot_cascade(self):
        """
        A cascade of CNOTs that equals identity.
        """

        qr = QuantumRegister(10, "qr")
        qc = QuantumCircuit(qr)
        qc.cx(qr[0], qr[1])
        qc.cx(qr[1], qr[2])
        qc.cx(qr[2], qr[3])
        qc.cx(qr[3], qr[4])
        qc.cx(qr[4], qr[5])
        qc.cx(qr[5], qr[6])
        qc.cx(qr[6], qr[7])
        qc.cx(qr[7], qr[8])
        qc.cx(qr[8], qr[9])

        qc.cx(qr[8], qr[9])
        qc.cx(qr[7], qr[8])
        qc.cx(qr[6], qr[7])
        qc.cx(qr[5], qr[6])
        qc.cx(qr[4], qr[5])
        qc.cx(qr[3], qr[4])
        qc.cx(qr[2], qr[3])
        qc.cx(qr[1], qr[2])
        qc.cx(qr[0], qr[1])

        qct = CommutativeOptimization()(qc)

        # The pass should cancel all of the gates in one go.
        expected = QuantumCircuit(qr)

        self.assertEqual(qct, expected)

    def test_cnot_cascade1(self):
        """
        A cascade of CNOTs that equals identity, with rotation gates inserted.
        """

        qr = QuantumRegister(10, "qr")
        qc = QuantumCircuit(qr)
        qc.rx(np.pi, qr[0])
        qc.rx(np.pi, qr[1])
        qc.rx(np.pi, qr[2])
        qc.rx(np.pi, qr[3])
        qc.rx(np.pi, qr[4])
        qc.rx(np.pi, qr[5])
        qc.rx(np.pi, qr[6])
        qc.rx(np.pi, qr[7])
        qc.rx(np.pi, qr[8])
        qc.rx(np.pi, qr[9])
        qc.cx(qr[0], qr[1])
        qc.cx(qr[1], qr[2])
        qc.cx(qr[2], qr[3])
        qc.cx(qr[3], qr[4])
        qc.cx(qr[4], qr[5])
        qc.cx(qr[5], qr[6])
        qc.cx(qr[6], qr[7])
        qc.cx(qr[7], qr[8])
        qc.cx(qr[8], qr[9])
        qc.cx(qr[8], qr[9])
        qc.cx(qr[7], qr[8])
        qc.cx(qr[6], qr[7])
        qc.cx(qr[5], qr[6])
        qc.cx(qr[4], qr[5])
        qc.cx(qr[3], qr[4])
        qc.cx(qr[2], qr[3])
        qc.cx(qr[1], qr[2])
        qc.cx(qr[0], qr[1])
        qc.rx(np.pi, qr[0])
        qc.rx(np.pi, qr[1])
        qc.rx(np.pi, qr[2])
        qc.rx(np.pi, qr[3])
        qc.rx(np.pi, qr[4])
        qc.rx(np.pi, qr[5])
        qc.rx(np.pi, qr[6])
        qc.rx(np.pi, qr[7])
        qc.rx(np.pi, qr[8])
        qc.rx(np.pi, qr[9])

        qct = CommutativeOptimization()(qc)

        # The pass should cancel all of the gates in one go.
        expected = QuantumCircuit(qr)

        self.assertEqual(qct, expected)

    def test_merge_rz_rotations1(self):
        """Test merging RZ-rotations."""
        qc = QuantumCircuit(1)
        qc.s(0)
        qc.z(0)
        qc.t(0)
        qc.rz(np.pi, 0)

        qct = CommutativeOptimization()(qc)

        expected = QuantumCircuit(1)
        expected.rz(11 * np.pi / 4, 0)
        expected.global_phase = 11 * np.pi / 4 / 2 - np.pi / 2

        self.assertEqual(Operator(expected), Operator(qc))
        self.assertEqual(qct, expected)

    def test_merge_rz_rotations2(self):
        """Test merging RZ-rotations."""
        qc = QuantumCircuit(1)
        qc.s(0)
        qc.z(0)
        qc.t(0)

        qct = CommutativeOptimization()(qc)

        expected = QuantumCircuit(1)
        expected.rz(7 * np.pi / 4, 0)
        expected.global_phase = 7 * np.pi / 4 / 2

        self.assertEqual(Operator(expected), Operator(qc))
        self.assertEqual(qct, expected)

    def test_simple_if_else(self):
        """Test that the pass is not confused by if-else."""
        base_test1 = QuantumCircuit(3, 3)
        base_test1.x(1)
        base_test1.cx(0, 1)
        base_test1.x(1)

        base_test2 = QuantumCircuit(3, 3)
        base_test2.rz(0.1, 1)
        base_test2.rz(0.1, 1)

        test = QuantumCircuit(3, 3)
        test.h(0)
        test.x(0)
        test.rx(0.2, 0)
        test.measure(0, 0)
        test.x(0)
        test.if_else(
            (test.clbits[0], True), base_test1.copy(), base_test2.copy(), test.qubits, test.clbits
        )

        expected = QuantumCircuit(3, 3, global_phase=np.pi)
        expected.h(0)
        expected.rx(np.pi + 0.2, 0)  # transforming X into RX(pi) introduces a pi/2 global phase
        expected.measure(0, 0)
        expected.rx(np.pi, 0)

        expected_test1 = QuantumCircuit(3, 3)
        expected_test1.cx(0, 1)

        expected_test2 = QuantumCircuit(3, 3)
        expected_test2.rz(0.2, 1)

        expected = QuantumCircuit(3, 3, global_phase=np.pi / 2)
        expected.h(0)
        expected.rx(np.pi + 0.2, 0)  # transforming X into RX(pi) introduces a pi/2 global phase
        expected.measure(0, 0)
        expected.x(0)

        expected_test1 = QuantumCircuit(3, 3)
        expected_test1.cx(0, 1)

        expected_test2 = QuantumCircuit(3, 3)
        expected_test2.rz(0.2, 1)

        expected.if_else(
            (expected.clbits[0], True),
            expected_test1.copy(),
            expected_test2.copy(),
            expected.qubits,
            expected.clbits,
        )

        qct = CommutativeOptimization()(test)
        self.assertEqual(qct, expected)

    def test_nested_control_flow(self):
        """Test that the pass does not add barrier into nested control flow."""
        level2_test = QuantumCircuit(2, 1)
        level2_test.cz(0, 1)
        level2_test.cz(0, 1)
        level2_test.cz(0, 1)
        level2_test.measure(0, 0)

        level1_test = QuantumCircuit(2, 1)
        level1_test.for_loop((0,), None, level2_test.copy(), level1_test.qubits, level1_test.clbits)
        level1_test.h(0)
        level1_test.h(0)
        level1_test.measure(0, 0)

        test = QuantumCircuit(2, 1)
        test.while_loop((test.clbits[0], True), level1_test.copy(), test.qubits, test.clbits)
        test.measure(0, 0)

        level2_expected = QuantumCircuit(2, 1)
        level2_expected.cz(0, 1)
        level2_expected.measure(0, 0)

        level1_expected = QuantumCircuit(2, 1)
        level1_expected.for_loop(
            (0,), None, level2_expected.copy(), level1_expected.qubits, level1_expected.clbits
        )
        level1_expected.measure(0, 0)

        expected = QuantumCircuit(2, 1)
        expected.while_loop(
            (expected.clbits[0], True), level1_expected.copy(), expected.qubits, expected.clbits
        )
        expected.measure(0, 0)

        qct = CommutativeOptimization()(test)
        self.assertEqual(qct, expected)

    def test_cancellation_not_crossing_block_boundary(self):
        """Test that the pass does cancel gates across control flow op block boundaries."""
        test1 = QuantumCircuit(2, 2)
        test1.x(1)
        with test1.if_test((0, False)):
            test1.cx(0, 1)
            test1.x(1)

        qct = CommutativeOptimization()(test1)

        self.assertEqual(qct, test1)

    def test_cancellation_not_crossing_between_blocks(self):
        """Test that the pass does cancel gates in different control flow ops."""
        test2 = QuantumCircuit(2, 2)
        with test2.if_test((0, True)):
            test2.x(1)
        with test2.if_test((0, True)):
            test2.cx(0, 1)
            test2.x(1)

        qct = CommutativeOptimization()(test2)
        self.assertEqual(qct, test2)

    def test_no_intransitive_cancellation(self):
        """Test that no unsound optimization occurs due to "intransitively-commuting" gates.
        See: https://github.com/Qiskit/qiskit-terra/issues/8020.
        """
        qc = QuantumCircuit(1)

        qc.x(0)
        qc.id(0)
        qc.h(0)
        qc.id(0)
        qc.x(0)

        qct = CommutativeOptimization()(qc)
        self.assertEqual(Operator(qc), Operator(qct))

    def test_overloaded_standard_gate_name(self):
        """Validate the pass works with custom gates using overloaded names

        See: https://github.com/Qiskit/qiskit/issues/13988 for more details.
        """
        qasm_str = """OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1
{
 rx(pi/2) q0;
 rx(pi/2) q1;
 cx q0,q1;
 rz(0.37801308) q1;
 cx q0,q1;
 rx(-pi/2) q0;
 rx(-pi/2) q1;
}
qreg q0[2];
creg c0[2];
z q0[0];
ryy(1.2182379) q0[0],q0[1];
z q0[0];
measure q0[0] -> c0[0];
measure q0[1] -> c0[1];
"""
        qc = QuantumCircuit.from_qasm_str(qasm_str)
        qct = CommutativeOptimization()(qc)
        # We don't cancel any gates with a custom rzz gate
        self.assertEqual(qct.count_ops()["z"], 2)

    def test_determinism(self):
        """Test that the pass produces structurally equivalent circuits."""
        # This is two CZ rings in a row.  If the cancellation order is non-deterministic and each
        # order has an equal chance, the probability of a spurious pass is astronoomical; the edge
        # IDs linking the in- and out-nodes will be different.
        qc = QuantumCircuit(21)
        for _ in range(2):
            for a, b in zip(qc.qubits[:-1], qc.qubits[1:]):
                qc.cz(a, b)
            qc.cz(qc.qubits[-1], qc.qubits[0])

        expected = circuit_to_dag(qc.copy_empty_like())

        left = CommutativeOptimization().run(circuit_to_dag(qc))
        right = CommutativeOptimization().run(circuit_to_dag(qc))

        # Semantic sanity checks.
        self.assertEqual(expected, left)
        self.assertEqual(expected, right)

        # The actual asseertion.
        self.assertTrue(left.structurally_equal(right))


if __name__ == "__main__":
    unittest.main()
