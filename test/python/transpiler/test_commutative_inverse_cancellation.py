# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test transpiler pass that cancels inverse gates while exploiting the commutation relations."""

import unittest
import numpy as np
from qiskit.test import QiskitTestCase

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import RZGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutativeInverseCancellation


class TestCommutativeInverseCancellation(QiskitTestCase):
    """Test the CommutativeInverseCancellation pass."""

    # The first suite of tests is adapted from CommutativeCancellation,
    # excluding/modifying the tests the combine rotations gates or do
    # basis priority change.

    def test_commutative_circuit1(self):
        """A simple circuit where three CNOTs commute, the first and the last cancel.

        0:----.---------------.--       0:------------
              |               |
        1:---(+)-----(+)-----(+)-   =   1:-------(+)--
                      |                           |
        2:---[H]------.----------       2:---[H]--.---
        """
        circuit = QuantumCircuit(3)
        circuit.cx(0, 1)
        circuit.h(2)
        circuit.cx(2, 1)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(3)
        expected.h(2)
        expected.cx(2, 1)

        self.assertEqual(expected, new_circuit)

    def test_consecutive_cnots(self):
        """A simple circuit equals identity

        0:----.- ----.--       0:------------
              |      |
        1:---(+)----(+)-   =   1:------------
        """

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)

        self.assertEqual(expected, new_circuit)

    def test_consecutive_cnots2(self):
        """
        Both CNOTs and rotations should cancel out.
        """
        circuit = QuantumCircuit(2)
        circuit.rx(np.pi / 2, 0)
        circuit.cx(0, 1)
        circuit.cx(0, 1)
        circuit.rx(-np.pi / 2, 0)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        self.assertEqual(expected, new_circuit)

    def test_2_alternating_cnots(self):
        """A simple circuit where nothing should be cancelled.

        0:----.- ---(+)-       0:----.----(+)-
              |      |               |     |
        1:---(+)-----.--   =   1:---(+)----.--

        """

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.cx(1, 0)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        expected.cx(0, 1)
        expected.cx(1, 0)

        self.assertEqual(expected, new_circuit)

    def test_control_bit_of_cnot(self):
        """A simple circuit where nothing should be cancelled.

        0:----.------[X]------.--       0:----.------[X]------.--
              |               |               |               |
        1:---(+)-------------(+)-   =   1:---(+)-------------(+)-
        """

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.x(0)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        expected.cx(0, 1)
        expected.x(0)
        expected.cx(0, 1)

        self.assertEqual(expected, new_circuit)

    def test_control_bit_of_cnot1(self):
        """A simple circuit where the two cnots should be cancelled.

        0:----.------[Z]------.--       0:---[Z]---
              |               |
        1:---(+)-------------(+)-   =   1:---------
        """

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.z(0)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        expected.z(0)

        self.assertEqual(expected, new_circuit)

    def test_control_bit_of_cnot2(self):
        """A simple circuit where the two cnots should be cancelled.

        0:----.------[T]------.--       0:---[T]---
              |               |
        1:---(+)-------------(+)-   =   1:---------
        """

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.t(0)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        expected.t(0)

        self.assertEqual(expected, new_circuit)

    def test_control_bit_of_cnot3(self):
        """A simple circuit where the two cnots should be cancelled.

        0:----.------[Rz]------.--       0:---[Rz]---
              |                |
        1:---(+)--------------(+)-   =   1:----------
        """

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.rz(np.pi / 3, 0)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        expected.rz(np.pi / 3, 0)

        self.assertEqual(expected, new_circuit)

    def test_control_bit_of_cnot4(self):
        """A simple circuit where the two cnots should be cancelled.

        0:----.------[T]------.--       0:---[T]---
              |               |
        1:---(+)-------------(+)-   =   1:---------
        """

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.t(0)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        expected.t(0)

        self.assertEqual(expected, new_circuit)

    def test_target_bit_of_cnot(self):
        """A simple circuit where nothing should be cancelled.

        0:----.---------------.--       0:----.---------------.--
              |               |               |               |
        1:---(+)-----[Z]-----(+)-   =   1:---(+)----[Z]------(+)-
        """

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.z(1)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        expected.cx(0, 1)
        expected.z(1)
        expected.cx(0, 1)

        self.assertEqual(expected, new_circuit)

    def test_target_bit_of_cnot1(self):
        """A simple circuit where nothing should be cancelled.

        0:----.---------------.--       0:----.---------------.--
              |               |               |               |
        1:---(+)-----[T]-----(+)-   =   1:---(+)----[T]------(+)-
        """

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.t(1)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        expected.cx(0, 1)
        expected.t(1)
        expected.cx(0, 1)

        self.assertEqual(expected, new_circuit)

    def test_target_bit_of_cnot2(self):
        """A simple circuit where nothing should be cancelled.

        0:----.---------------.--       0:----.---------------.--
              |               |               |               |
        1:---(+)-----[Rz]----(+)-   =   1:---(+)----[Rz]-----(+)-
        """

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.rz(np.pi / 3, 1)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        expected.cx(0, 1)
        expected.rz(np.pi / 3, 1)
        expected.cx(0, 1)

        self.assertEqual(expected, new_circuit)

    def test_commutative_circuit2(self):
        """
        A simple circuit where three CNOTs commute, the first and the last cancel,
        also two X gates cancel.
        """

        circuit = QuantumCircuit(3)
        circuit.cx(0, 1)
        circuit.rz(np.pi / 3, 2)
        circuit.cx(2, 1)
        circuit.rz(np.pi / 3, 2)
        circuit.t(2)
        circuit.s(2)
        circuit.x(1)
        circuit.cx(0, 1)
        circuit.x(1)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(3)
        expected.rz(np.pi / 3, 2)
        expected.cx(2, 1)
        expected.rz(np.pi / 3, 2)
        expected.t(2)
        expected.s(2)

        self.assertEqual(expected, new_circuit)

    def test_commutative_circuit3(self):
        """
        A simple circuit where three CNOTs commute, the first and the last cancel,
        also two X gates cancel and two RX gates cancel.
        """

        circuit = QuantumCircuit(4)

        circuit.cx(0, 1)
        circuit.rz(np.pi / 3, 2)
        circuit.rz(np.pi / 3, 3)
        circuit.x(3)
        circuit.cx(2, 3)
        circuit.cx(2, 1)
        circuit.cx(2, 3)
        circuit.rz(-np.pi / 3, 2)
        circuit.x(3)
        circuit.rz(-np.pi / 3, 3)
        circuit.x(1)
        circuit.cx(0, 1)
        circuit.x(1)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(4)
        expected.cx(2, 1)

        self.assertEqual(expected, new_circuit)

    def test_cnot_cascade(self):
        """
        A cascade of CNOTs that equals identity.
        """

        circuit = QuantumCircuit(10)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(2, 3)
        circuit.cx(3, 4)
        circuit.cx(4, 5)
        circuit.cx(5, 6)
        circuit.cx(6, 7)
        circuit.cx(7, 8)
        circuit.cx(8, 9)

        circuit.cx(8, 9)
        circuit.cx(7, 8)
        circuit.cx(6, 7)
        circuit.cx(5, 6)
        circuit.cx(4, 5)
        circuit.cx(3, 4)
        circuit.cx(2, 3)
        circuit.cx(1, 2)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(10)

        self.assertEqual(expected, new_circuit)

    def test_conditional_gates_dont_commute(self):
        """Conditional gates do not commute and do not cancel"""

        #      ┌───┐┌─┐
        # q_0: ┤ H ├┤M├─────────────
        #      └───┘└╥┘       ┌─┐
        # q_1: ──■───╫────■───┤M├───
        #      ┌─┴─┐ ║  ┌─┴─┐ └╥┘┌─┐
        # q_2: ┤ X ├─╫──┤ X ├──╫─┤M├
        #      └───┘ ║  └─╥─┘  ║ └╥┘
        #            ║ ┌──╨──┐ ║  ║
        # c: 2/══════╩═╡ 0x0 ╞═╩══╩═
        #            0 └─────┘ 0  1
        circuit = QuantumCircuit(3, 2)
        circuit.h(0)
        circuit.measure(0, 0)
        circuit.cx(1, 2)
        circuit.cx(1, 2).c_if(circuit.cregs[0], 0)
        circuit.measure([1, 2], [0, 1])

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)

        self.assertEqual(circuit, new_circuit)

    # The second suite of tests is adapted from InverseCancellation,
    # modifying tests where more nonconsecutive gates cancel.

    def test_basic_self_inverse(self):
        """Test that a single self-inverse gate as input can be cancelled."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.h(0)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertNotIn("h", gates_after)

    def test_odd_number_self_inverse(self):
        """Test that an odd number of self-inverse gates leaves one gate remaining."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.h(0)
        circuit.h(0)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertIn("h", gates_after)
        self.assertEqual(gates_after["h"], 1)

    def test_basic_cx_self_inverse(self):
        """Test that a single self-inverse cx gate as input can be cancelled."""
        circuit = QuantumCircuit(2, 2)
        circuit.cx(0, 1)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertNotIn("cx", gates_after)

    def test_basic_gate_inverse(self):
        """Test that a basic pair of gate inverse can be cancelled."""
        circuit = QuantumCircuit(2, 2)
        circuit.rx(np.pi / 4, 0)
        circuit.rx(-np.pi / 4, 0)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertNotIn("rx", gates_after)

    def test_non_inverse_do_not_cancel(self):
        """Test that non-inverse gate pairs do not cancel."""
        circuit = QuantumCircuit(2, 2)
        circuit.rx(np.pi / 4, 0)
        circuit.rx(np.pi / 4, 0)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertIn("rx", gates_after)
        self.assertEqual(gates_after["rx"], 2)

    def test_non_consecutive_gates(self):
        """Test that non-consecutive gates cancel as well."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.h(0)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(0, 1)
        circuit.h(0)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertNotIn("cx", gates_after)
        self.assertNotIn("h", gates_after)

    def test_gate_inverse_phase_gate(self):
        """Test that an inverse pair of a PhaseGate can be cancelled."""
        circuit = QuantumCircuit(2, 2)
        circuit.p(np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertNotIn("p", gates_after)

    def test_self_inverse_on_different_qubits(self):
        """Test that self_inverse gates cancel on the correct qubits."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.h(1)
        circuit.h(0)
        circuit.h(1)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertNotIn("h", gates_after)

    def test_consecutive_self_inverse_h_x_gate(self):
        """Test that consecutive self-inverse gates cancel."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.h(0)
        circuit.h(0)
        circuit.x(0)
        circuit.x(0)
        circuit.h(0)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertNotIn("x", gates_after)
        self.assertNotIn("h", gates_after)

    def test_inverse_with_different_names(self):
        """Test that inverse gates that have different names."""
        circuit = QuantumCircuit(2, 2)
        circuit.t(0)
        circuit.tdg(0)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertNotIn("t", gates_after)
        self.assertNotIn("tdg", gates_after)

    def test_three_alternating_inverse_gates(self):
        """Test that inverse cancellation works correctly for alternating sequences
        of inverse gates of odd-length."""
        circuit = QuantumCircuit(2, 2)
        circuit.p(np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)
        circuit.p(np.pi / 4, 0)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertIn("p", gates_after)
        self.assertEqual(gates_after["p"], 1)

    def test_four_alternating_inverse_gates(self):
        """Test that inverse cancellation works correctly for alternating sequences
        of inverse gates of even-length."""
        circuit = QuantumCircuit(2, 2)
        circuit.p(np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)
        circuit.p(np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertNotIn("p", gates_after)

    def test_five_alternating_inverse_gates(self):
        """Test that inverse cancellation works correctly for alternating sequences
        of inverse gates of odd-length."""
        circuit = QuantumCircuit(2, 2)
        circuit.p(np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)
        circuit.p(np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)
        circuit.p(np.pi / 4, 0)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertIn("p", gates_after)
        self.assertEqual(gates_after["p"], 1)

    def test_sequence_of_inverse_gates_1(self):
        """Test that inverse cancellation works correctly for more general sequences
        of inverse gates. In this test two pairs of inverse gates are supposed to
        cancel out."""
        circuit = QuantumCircuit(2, 2)
        circuit.p(np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)
        circuit.p(np.pi / 4, 0)
        circuit.p(np.pi / 4, 0)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertIn("p", gates_after)
        self.assertEqual(gates_after["p"], 1)

    def test_sequence_of_inverse_gates_2(self):
        """Test that inverse cancellation works correctly for more general sequences
        of inverse gates. In this test, in theory three pairs of inverse gates can
        cancel out, but in practice only two pairs are back-to-back."""
        circuit = QuantumCircuit(2, 2)
        circuit.p(np.pi / 4, 0)
        circuit.p(np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)
        circuit.p(-np.pi / 4, 0)
        circuit.p(np.pi / 4, 0)
        circuit.p(np.pi / 4, 0)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertIn("p", gates_after)
        self.assertEqual(gates_after["p"] % 2, 1)

    def test_cx_do_not_wrongly_cancel(self):
        """Test that CX(0,1) and CX(1, 0) do not cancel out, when (CX, CX) is passed
        as an inverse pair."""
        circuit = QuantumCircuit(2, 0)
        circuit.cx(0, 1)
        circuit.cx(1, 0)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)
        gates_after = new_circuit.count_ops()

        self.assertIn("cx", gates_after)
        self.assertEqual(gates_after["cx"], 2)

    # A few more tests from issue 8020

    def test_cancel_both_x_and_z(self):
        """Test that Z commutes with control qubit of CX, and X commutes with the target qubit."""
        circuit = QuantumCircuit(2)
        circuit.z(0)
        circuit.x(1)
        circuit.cx(0, 1)
        circuit.z(0)
        circuit.x(1)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(2)
        expected.cx(0, 1)

        self.assertEqual(expected, new_circuit)

    def test_gates_do_not_wrongly_cancel(self):
        """Test that X gates do not cancel for X-I-H-I-X."""
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.i(0)
        circuit.h(0)
        circuit.i(0)
        circuit.x(0)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(1)
        expected.x(0)
        expected.h(0)
        expected.x(0)

        self.assertEqual(expected, new_circuit)

    # More tests to cover corner-cases: parameterized gates, directives, reset, etc.

    def test_no_cancellation_across_barrier(self):
        """Test that barrier prevents cancellation."""
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.barrier()
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)

        self.assertEqual(circuit, new_circuit)

    def test_no_cancellation_across_measure(self):
        """Test that barrier prevents cancellation."""
        circuit = QuantumCircuit(2, 1)
        circuit.cx(0, 1)
        circuit.measure(0, 0)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)

        self.assertEqual(circuit, new_circuit)

    def test_no_cancellation_across_reset(self):
        """Test that reset prevents cancellation."""
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.reset(0)
        circuit.cx(0, 1)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)

        self.assertEqual(circuit, new_circuit)

    def test_no_cancellation_across_parameterized_gates(self):
        """Test that parameterized gates prevent cancellation.
        This test should be modified when inverse and commutativity checking
        get improved to handle parameterized gates.
        """
        circuit = QuantumCircuit(1)
        circuit.rz(np.pi / 2, 0)
        circuit.rz(Parameter("Theta"), 0)
        circuit.rz(-np.pi / 2, 0)

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)
        self.assertEqual(circuit, new_circuit)

    def test_parameterized_gates_do_not_cancel(self):
        """Test that parameterized gates do not cancel.
        This test should be modified when inverse and commutativity checking
        get improved to handle parameterized gates.
        """
        gate = RZGate(Parameter("Theta"))

        circuit = QuantumCircuit(1)
        circuit.append(gate, [0])
        circuit.append(gate.inverse(), [0])

        passmanager = PassManager(CommutativeInverseCancellation())
        new_circuit = passmanager.run(circuit)
        self.assertEqual(circuit, new_circuit)


if __name__ == "__main__":
    unittest.main()
