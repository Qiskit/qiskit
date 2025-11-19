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
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZGate
from qiskit.transpiler.passes import ControlPatternSimplification
from qiskit.quantum_info import Statevector, state_fidelity
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from qiskit.utils import optionals


class TestControlPatternSimplification(QiskitTestCase):
    """Tests for ControlPatternSimplification transpiler pass."""

    def _verify_all_states_fidelity(self, qc, optimized_qc, num_qubits):
        """Helper method to verify state fidelity across all basis states.

        Args:
            qc: Original quantum circuit
            optimized_qc: Optimized quantum circuit
            num_qubits: Number of qubits in the circuit
        """
        # Test all 2^n basis states
        for i in range(2**num_qubits):
            # Prepare basis state |i⟩
            basis_state = QuantumCircuit(num_qubits)
            for j, bit in enumerate(format(i, f"0{num_qubits}b")):
                if bit == "1":
                    basis_state.x(j)

            # Apply original circuit
            test_qc_original = basis_state.compose(qc)
            original_sv = Statevector.from_instruction(test_qc_original)

            # Apply optimized circuit
            test_qc_optimized = basis_state.copy()
            test_qc_optimized = test_qc_optimized.compose(optimized_qc)
            optimized_sv = Statevector.from_instruction(test_qc_optimized)

            # Verify fidelity for this basis state
            fidelity = state_fidelity(original_sv, optimized_sv)
            self.assertAlmostEqual(
                fidelity,
                1.0,
                places=10,
                msg=f"Fidelity mismatch for basis state |{format(i, f'0{num_qubits}b')}⟩",
            )

    @unittest.skipUnless(optionals.HAS_SYMPY, "SymPy required for this test")
    def test_complementary_patterns_rx(self):
        """Test complementary control patterns with RX gates ('11' and '01' -> single control on q0)."""
        # Expected: 2 MCRX gates -> 1 CRX gate
        theta = np.pi / 4
        qc = QuantumCircuit(3)
        qc.append(RXGate(theta).control(2, ctrl_state="11"), [0, 1, 2])
        qc.append(RXGate(theta).control(2, ctrl_state="01"), [0, 1, 2])

        # Apply optimization pass
        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(qc)

        # Verify state equivalence across all basis states
        self._verify_all_states_fidelity(qc, optimized_qc, 3)

        # Verify gate count reduction
        original_count = sum(
            1 for instr in qc.data if isinstance(instr.operation, type(qc.data[0].operation))
        )
        optimized_count = sum(
            1
            for instr in optimized_qc.data
            if hasattr(instr.operation, "num_ctrl_qubits") and instr.operation.num_ctrl_qubits > 0
        )

        # Should reduce from 2 gates to 1 gate
        self.assertLess(
            optimized_count, original_count, msg="Optimized circuit should have fewer gates"
        )

    @unittest.skipUnless(optionals.HAS_SYMPY, "SymPy required for this test")
    def test_complementary_patterns_ry(self):
        """Test complementary control patterns with RY gates."""
        # Verify gate-agnostic optimization works for RY
        theta = np.pi / 4
        qc = QuantumCircuit(3)
        qc.append(RYGate(theta).control(2, ctrl_state="11"), [0, 1, 2])
        qc.append(RYGate(theta).control(2, ctrl_state="01"), [0, 1, 2])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(qc)

        # Verify state equivalence across all basis states
        self._verify_all_states_fidelity(qc, optimized_qc, 3)

    @unittest.skipUnless(optionals.HAS_SYMPY, "SymPy required for this test")
    def test_complementary_patterns_rz(self):
        """Test complementary control patterns with RZ gates."""
        # Verify gate-agnostic optimization works for RZ
        theta = np.pi / 4
        qc = QuantumCircuit(3)
        qc.append(RZGate(theta).control(2, ctrl_state="11"), [0, 1, 2])
        qc.append(RZGate(theta).control(2, ctrl_state="01"), [0, 1, 2])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(qc)

        # Verify state equivalence across all basis states
        self._verify_all_states_fidelity(qc, optimized_qc, 3)

    @unittest.skipUnless(optionals.HAS_SYMPY, "SymPy required for this test")
    def test_subset_patterns(self):
        """Test subset control patterns ('111' and '110' -> reduce control count)."""
        # Patterns where q0∧q1∧q2 ∨ q0∧q1∧¬q2 = q0∧q1
        theta = np.pi / 3
        qc = QuantumCircuit(4)
        qc.append(RXGate(theta).control(3, ctrl_state="111"), [0, 1, 2, 3])
        qc.append(RXGate(theta).control(3, ctrl_state="110"), [0, 1, 2, 3])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(qc)

        # Verify state equivalence across all basis states
        self._verify_all_states_fidelity(qc, optimized_qc, 4)

    @unittest.skipUnless(optionals.HAS_SYMPY, "SymPy required for this test")
    def test_complete_partition(self):
        """Test complete partition patterns (['00','01','10','11'] -> unconditional)."""
        # All control states covered -> unconditional gate
        theta = np.pi / 6
        qc = QuantumCircuit(3)
        qc.append(RXGate(theta).control(2, ctrl_state="00"), [0, 1, 2])
        qc.append(RXGate(theta).control(2, ctrl_state="01"), [0, 1, 2])
        qc.append(RXGate(theta).control(2, ctrl_state="10"), [0, 1, 2])
        qc.append(RXGate(theta).control(2, ctrl_state="11"), [0, 1, 2])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(qc)

        # Verify state equivalence across all basis states
        self._verify_all_states_fidelity(qc, optimized_qc, 3)

    @unittest.skipUnless(optionals.HAS_SYMPY, "SymPy required for this test")
    def test_no_optimization_different_params(self):
        """Test that gates with different parameters are not optimized together."""
        theta1 = np.pi / 4
        theta2 = np.pi / 3
        qc = QuantumCircuit(3)
        qc.append(RXGate(theta1).control(2, ctrl_state="11"), [0, 1, 2])
        qc.append(RXGate(theta2).control(2, ctrl_state="01"), [0, 1, 2])  # Different angle

        original_count = len([op for op in qc.data])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(qc)

        optimized_count = len([op for op in optimized_qc.data])

        # Should NOT optimize due to different parameters
        self.assertEqual(original_count, optimized_count)

        # Verify state equivalence across all basis states
        self._verify_all_states_fidelity(qc, optimized_qc, 3)

    @unittest.skipUnless(optionals.HAS_SYMPY, "SymPy required for this test")
    def test_identical_patterns_different_params_merge(self):
        """Test that gates with identical patterns and different angles can merge."""
        theta1 = np.pi / 4
        theta2 = np.pi / 3
        qc = QuantumCircuit(3)
        qc.append(RXGate(theta1).control(2, ctrl_state="11"), [0, 1, 2])
        qc.append(RXGate(theta2).control(2, ctrl_state="11"), [0, 1, 2])  # Same pattern, different angle

        original_count = len([op for op in qc.data])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(qc)

        optimized_count = len([op for op in optimized_qc.data])

        # Gates with same pattern can be merged (angles add up)
        # This is left to other optimization passes, so we just verify equivalence
        self.assertLessEqual(optimized_count, original_count)

        # Verify state equivalence across all basis states
        self._verify_all_states_fidelity(qc, optimized_qc, 3)

    @unittest.skipUnless(optionals.HAS_SYMPY, "SymPy required for this test")
    def test_comprehensive_state_fidelity(self):
        """Test state equivalence across all basis states."""
        theta = np.pi / 5
        qc = QuantumCircuit(3)
        qc.append(RXGate(theta).control(2, ctrl_state="11"), [0, 1, 2])
        qc.append(RXGate(theta).control(2, ctrl_state="01"), [0, 1, 2])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(qc)

        # Verify state equivalence across all basis states
        self._verify_all_states_fidelity(qc, optimized_qc, 3)

    @unittest.skipUnless(optionals.HAS_SYMPY, "SymPy required for this test")
    def test_three_control_complementary(self):
        """Test complementary patterns with 3 control qubits ('111' and '011')."""
        # Pattern: (q0 & q1 & q2) | (~q0 & q1 & q2) = q1 & q2
        theta = np.pi / 6
        qc = QuantumCircuit(4)
        qc.append(RXGate(theta).control(3, ctrl_state="111"), [0, 1, 2, 3])
        qc.append(RXGate(theta).control(3, ctrl_state="011"), [0, 1, 2, 3])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(qc)

        # Verify state equivalence across all basis states
        self._verify_all_states_fidelity(qc, optimized_qc, 4)

        # Should reduce from 2 3-control gates to 1 2-control gate
        original_ctrl_count = sum(
            1
            for instr in qc.data
            if hasattr(instr.operation, "num_ctrl_qubits") and instr.operation.num_ctrl_qubits == 3
        )
        optimized_ctrl_count = sum(
            1
            for instr in optimized_qc.data
            if hasattr(instr.operation, "num_ctrl_qubits") and instr.operation.num_ctrl_qubits >= 2
        )

        # Original has 2 gates, optimized should have 1 gate with fewer controls
        self.assertEqual(original_ctrl_count, 2)
        self.assertLessEqual(optimized_ctrl_count, 1)

    @unittest.skipUnless(optionals.HAS_SYMPY, "SymPy required for this test")
    def test_multiple_patterns_same_result(self):
        """Test 4 patterns that simplify to 2-control: '1110', '1100', '1111', '1101'."""
        # All have q0=1, q1=1 in common
        theta = np.pi / 7
        qc = QuantumCircuit(5)
        qc.append(RXGate(theta).control(4, ctrl_state="1110"), [0, 1, 2, 3, 4])
        qc.append(RXGate(theta).control(4, ctrl_state="1100"), [0, 1, 2, 3, 4])
        qc.append(RXGate(theta).control(4, ctrl_state="1111"), [0, 1, 2, 3, 4])
        qc.append(RXGate(theta).control(4, ctrl_state="1101"), [0, 1, 2, 3, 4])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(qc)

        # Verify state equivalence across all basis states
        self._verify_all_states_fidelity(qc, optimized_qc, 5)

    @unittest.skipUnless(optionals.HAS_SYMPY, "SymPy required for this test")
    def test_inverted_control_patterns(self):
        """Test with inverted controls ('00' and '10')."""
        # Pattern: (~q0 & ~q1) | (q0 & ~q1) = ~q1
        theta = np.pi / 8
        qc = QuantumCircuit(3)
        qc.append(RXGate(theta).control(2, ctrl_state="00"), [0, 1, 2])
        qc.append(RXGate(theta).control(2, ctrl_state="10"), [0, 1, 2])

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(qc)

        # Verify state equivalence across all basis states
        self._verify_all_states_fidelity(qc, optimized_qc, 3)

    @unittest.skipUnless(optionals.HAS_SYMPY, "SymPy required for this test")
    def test_mixed_gate_types_no_optimization(self):
        """Test that different gate types are not optimized together."""
        theta = np.pi / 4
        qc = QuantumCircuit(3)
        qc.append(RXGate(theta).control(2, ctrl_state="11"), [0, 1, 2])
        qc.append(RYGate(theta).control(2, ctrl_state="01"), [0, 1, 2])  # Different gate type

        original_count = len(qc.data)

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(qc)

        optimized_count = len(optimized_qc.data)

        # Should NOT optimize due to different gate types
        self.assertEqual(original_count, optimized_count)

        # Verify state equivalence across all basis states
        self._verify_all_states_fidelity(qc, optimized_qc, 3)

    @unittest.skipUnless(optionals.HAS_SYMPY, "SymPy required for this test")
    def test_different_target_qubits_no_optimization(self):
        """Test that gates on different target qubits are not optimized together."""
        theta = np.pi / 4
        qc = QuantumCircuit(4)
        qc.append(RXGate(theta).control(2, ctrl_state="11"), [0, 1, 2])
        qc.append(RXGate(theta).control(2, ctrl_state="01"), [0, 1, 3])  # Different target

        original_count = len(qc.data)

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(qc)

        optimized_count = len(optimized_qc.data)

        # Should NOT optimize due to different targets
        self.assertEqual(original_count, optimized_count)

        # Verify state equivalence across all basis states
        self._verify_all_states_fidelity(qc, optimized_qc, 4)

    @unittest.skipUnless(optionals.HAS_SYMPY, "SymPy required for this test")
    def test_single_gate_no_change(self):
        """Test that a single controlled gate is not modified."""
        theta = np.pi / 4
        qc = QuantumCircuit(3)
        qc.append(RXGate(theta).control(2, ctrl_state="11"), [0, 1, 2])

        original_count = len(qc.data)

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(qc)

        optimized_count = len(optimized_qc.data)

        # Should remain unchanged
        self.assertEqual(original_count, optimized_count)

        # Verify state equivalence across all basis states
        self._verify_all_states_fidelity(qc, optimized_qc, 3)

    @unittest.skipUnless(optionals.HAS_SYMPY, "SymPy required for this test")
    def test_non_consecutive_gates_no_optimization(self):
        """Test that non-consecutive controlled gates are not optimized together."""
        theta = np.pi / 4
        qc = QuantumCircuit(3)
        qc.append(RXGate(theta).control(2, ctrl_state="11"), [0, 1, 2])
        qc.h(2)  # Non-controlled gate breaks the run
        qc.append(RXGate(theta).control(2, ctrl_state="01"), [0, 1, 2])

        original_count = sum(1 for instr in qc.data if hasattr(instr.operation, "num_ctrl_qubits"))

        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(qc)

        optimized_count = sum(
            1 for instr in optimized_qc.data if hasattr(instr.operation, "num_ctrl_qubits")
        )

        # Controlled gates should not be optimized (separated by H gate)
        self.assertEqual(original_count, optimized_count)

        # Verify state equivalence across all basis states
        self._verify_all_states_fidelity(qc, optimized_qc, 3)


if __name__ == "__main__":
    unittest.main()
