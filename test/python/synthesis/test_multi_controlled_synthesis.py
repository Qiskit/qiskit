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

"""Test synthesis algorithms for multi-controlled gates."""

import unittest
from test import combine
import numpy as np
from ddt import ddt, data

from qiskit.quantum_info import Operator
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.library import (
    XGate,
    RXGate,
    RYGate,
    RZGate,
    PhaseGate,
    YGate,
    ZGate,
    HGate,
    SGate,
    SdgGate,
    TGate,
    TdgGate,
    SXGate,
    SXdgGate,
    UGate,
)
from qiskit.synthesis.multi_controlled import (
    synth_mcx_n_dirty_i15,
    synth_mcx_n_clean_m15,
    synth_mcx_1_clean_b95,
    synth_mcx_gray_code,
    synth_mcx_noaux_v24,
    synth_c3x,
    synth_c4x,
)
from qiskit.circuit._utils import _compute_control_matrix
from qiskit.quantum_info.operators.operator_utils import _equal_with_ancillas, matrix_equal
from qiskit.transpiler import generate_preset_pass_manager

from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestMCXSynthesis(QiskitTestCase):
    """Test MCX synthesis methods."""

    @staticmethod
    def mcx_matrix(num_ctrl_qubits: int):
        """Return matrix for the MCX gate with the given number of control qubits."""
        base_mat = XGate().to_matrix()
        return _compute_control_matrix(base_mat, num_ctrl_qubits)

    def check_mcx_synthesis(
        self, num_ctrl_qubits: int, synthesized_circuit: QuantumCircuit, clean_ancillas: bool
    ):
        """Check correctness of a quantum circuit produced by an MCX synthesis algorithm, taking
        the additional ancilla qubits into account.

        This check is based on comparing the synthesized and the expected matrices and thus only
        works for synthesized circuits with up to about 10 qubits.

        Args:
            num_ctrl_qubits: the number of control qubits for the MCX gate.
            synthesized_circuit: the quantum circuit synthesizing the MCX gate.
            clean_ancillas: True if the algorithm uses clean ancilla qubits.

        Note: currently we do not have any MCX synthesis algorithms that use both clean and dirty
        ancilla qubits. When we do, we will need to extend this function.
        """
        original_op = Operator(self.mcx_matrix(num_ctrl_qubits))
        synthesized_op = Operator(synthesized_circuit)

        num_qubits_original = original_op._op_shape._num_qargs_l
        num_qubits_synthesized = synthesized_circuit.num_qubits

        expected_op = Operator(
            np.kron(np.eye(2 ** (num_qubits_synthesized - num_qubits_original)), original_op)
        )
        if clean_ancillas:
            ancilla_qubits = list(range(num_qubits_original, num_qubits_synthesized))
        else:
            ancilla_qubits = []

        result = _equal_with_ancillas(
            synthesized_op,
            expected_op,
            ancilla_qubits,
        )
        self.assertTrue(result)

    @data(1, 2, 3, 4, 5, 6)
    def test_mcx_n_dirty_i15(self, num_ctrl_qubits: int):
        """Test synth_mcx_n_dirty_i15 by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_mcx_n_dirty_i15(num_ctrl_qubits)
        self.check_mcx_synthesis(num_ctrl_qubits, synthesized_circuit, clean_ancillas=False)

    @data(3, 4, 5, 6)
    def test_mcx_n_clean_m15(self, num_ctrl_qubits: int):
        """Test synth_mcx_n_clean_m15 by comparing synthesized and expected matrices."""
        # Note: the method requires at least 3 control qubits
        synthesized_circuit = synth_mcx_n_clean_m15(num_ctrl_qubits)
        self.check_mcx_synthesis(num_ctrl_qubits, synthesized_circuit, clean_ancillas=True)

    @data(3, 4, 5, 6, 7, 8)
    def test_mcx_1_clean_b95(self, num_ctrl_qubits: int):
        """Test synth_mcx_1_clean_b95 by comparing synthesized and expected matrices."""
        # Note: the method requires at least 3 control qubits
        synthesized_circuit = synth_mcx_1_clean_b95(num_ctrl_qubits)
        self.check_mcx_synthesis(num_ctrl_qubits, synthesized_circuit, clean_ancillas=True)

    @data(3, 4, 5, 6, 7, 8)
    def test_mcx_gray_code(self, num_ctrl_qubits: int):
        """Test synth_mcx_gray_code by comparing synthesized and expected matrices."""
        # Note: the method requires at least 3 control qubits
        synthesized_circuit = synth_mcx_gray_code(num_ctrl_qubits)
        self.check_mcx_synthesis(num_ctrl_qubits, synthesized_circuit, clean_ancillas=False)

    @data(1, 2, 3, 4, 5, 6, 7, 8)
    def test_mcx_noaux_v24(self, num_ctrl_qubits: int):
        """Test synth_mcx_noaux_v24 by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_mcx_noaux_v24(num_ctrl_qubits)
        self.check_mcx_synthesis(num_ctrl_qubits, synthesized_circuit, clean_ancillas=False)

    def test_c3x(self):
        """Test synth_c3x by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_c3x()
        self.check_mcx_synthesis(3, synthesized_circuit, clean_ancillas=False)

    def test_c4x(self):
        """Test synth_c4x by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_c4x()
        self.check_mcx_synthesis(4, synthesized_circuit, clean_ancillas=False)

    @data(5, 10, 15)
    def test_mcx_n_dirty_i15_cx_count(self, num_ctrl_qubits: int):
        """Test synth_mcx_n_dirty_i15 bound on CX count."""
        synthesized_circuit = synth_mcx_n_dirty_i15(num_ctrl_qubits)
        pm = generate_preset_pass_manager(
            optimization_level=0, basis_gates=["u", "cx"], seed_transpiler=12345
        )
        transpiled_circuit = pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        self.assertLessEqual(cx_count, 8 * num_ctrl_qubits - 6)

    @data(5, 10, 15)
    def test_mcx_n_clean_m15_cx_count(self, num_ctrl_qubits: int):
        """Test synth_mcx_n_clean_m15 bound on CX count."""
        synthesized_circuit = synth_mcx_n_clean_m15(num_ctrl_qubits)
        pm = generate_preset_pass_manager(
            optimization_level=0, basis_gates=["u", "cx"], seed_transpiler=12345
        )
        transpiled_circuit = pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        self.assertLessEqual(cx_count, 6 * num_ctrl_qubits - 6)

    @data(5, 10, 15)
    def test_mcx_1_clean_b95_cx_count(self, num_ctrl_qubits: int):
        """Test synth_mcx_1_clean_b95 bound on CX count."""
        synthesized_circuit = synth_mcx_1_clean_b95(num_ctrl_qubits)
        pm = generate_preset_pass_manager(
            optimization_level=0, basis_gates=["u", "cx"], seed_transpiler=12345
        )
        transpiled_circuit = pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        self.assertLessEqual(cx_count, 16 * num_ctrl_qubits - 8)

    @data(5, 8, 10, 13, 15)
    def test_mcx_noaux_v24_cx_count(self, num_ctrl_qubits: int):
        """Test synth_mcx_noaux_v24 bound on CX count."""
        synthesized_circuit = synth_mcx_noaux_v24(num_ctrl_qubits)
        pm = generate_preset_pass_manager(
            optimization_level=0, basis_gates=["u", "cx"], seed_transpiler=12345
        )
        transpiled_circuit = pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The bound is based on an arithmetic progression of mcrz gates
        self.assertLessEqual(cx_count, 8 * num_ctrl_qubits**2 - 16 * num_ctrl_qubits)

    def test_c3x_cx_count(self):
        """Test synth_c3x bound on CX count."""
        synthesized_circuit = synth_c3x()
        pm = generate_preset_pass_manager(
            optimization_level=0, basis_gates=["u", "cx"], seed_transpiler=12345
        )
        transpiled_circuit = pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        self.assertLessEqual(cx_count, 14)

    def test_c4x_cx_count(self):
        """Test synth_c4x bound on CX count."""
        synthesized_circuit = synth_c4x()
        pm = generate_preset_pass_manager(
            optimization_level=0, basis_gates=["u", "cx"], seed_transpiler=12345
        )
        transpiled_circuit = pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        self.assertLessEqual(cx_count, 36)


@ddt
class TestMCSynthesis(QiskitTestCase):
    """Test multi-controlled synthesis methods."""

    @combine(
        num_ctrl_qubits=[5, 8, 10, 13, 15],
        base_gate=[
            XGate(),
            YGate(),
            ZGate(),
            HGate(),
            SGate(),
            SdgGate(),
            TGate(),
            TdgGate(),
            SXGate(),
            SXdgGate(),
            PhaseGate(0.345),
        ],
    )
    def test_mcx_equiv_noaux_cx_count(self, num_ctrl_qubits: int, base_gate: Gate):
        """Test gates which are locally equivalent to MCX synthesis bound on CX count."""
        qc = QuantumCircuit(num_ctrl_qubits + 1)
        qc.append(base_gate.control(num_ctrl_qubits), range(num_ctrl_qubits + 1))
        pm = generate_preset_pass_manager(
            optimization_level=0, basis_gates=["u", "cx"], seed_transpiler=12345
        )
        transpiled_circuit = pm.run(qc)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The bound is based on an arithmetic progression of mcrz gates
        self.assertLessEqual(cx_count, 8 * num_ctrl_qubits**2 - 16 * num_ctrl_qubits)

    @combine(
        num_ctrl_qubits=[4, 5, 8, 10, 13, 15],
        base_gate=[RXGate(0.123), RYGate(0.456), RZGate(0.789)],
    )
    def test_mc_rotation_gates_cx_count(self, num_ctrl_qubits: int, base_gate: Gate):
        """Test mcrx / mcry / mcrz synthesis bound on CX count."""
        qc = QuantumCircuit(num_ctrl_qubits + 1)
        qc.append(base_gate.control(num_ctrl_qubits), range(num_ctrl_qubits + 1))
        pm = generate_preset_pass_manager(
            optimization_level=0, basis_gates=["u", "cx"], seed_transpiler=12345
        )
        transpiled_circuit = pm.run(qc)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The bound is based on arXiv:2302.06377, Theorem 3
        self.assertLessEqual(cx_count, 16 * (num_ctrl_qubits + 1) - 40)

    @data(5, 8, 10, 13, 15)
    def test_mcu_noaux_cx_count(self, num_ctrl_qubits: int):
        """Test multi-controlled random unitary synthesis bound on CX count."""
        base_gate = UGate(0.123, 0.456, 0.789)
        qc = QuantumCircuit(num_ctrl_qubits + 1)
        qc.append(base_gate.control(num_ctrl_qubits), range(num_ctrl_qubits + 1))
        pm = generate_preset_pass_manager(
            optimization_level=0, basis_gates=["u", "cx"], seed_transpiler=12345
        )
        transpiled_circuit = pm.run(qc)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The bound is based on the synthesis of multi-controlled UGate using two mcrz gates,
        # one mcry gate and one mcp gate (with num_ctrl_qubits-1 controls)
        self.assertLessEqual(cx_count, 8 * num_ctrl_qubits**2 + 16 * num_ctrl_qubits - 96)

    @combine(num_ctrl_qubits=[1, 2, 3, 4], base_gate=[XGate(), YGate(), ZGate(), HGate()])
    def test_small_mcx_gates_yield_cx_count(self, num_ctrl_qubits, base_gate):
        """Test that creating a MCX gate (and other locally equivalent multi-controlled gates)
        with small number of controls (with no ancillas) yields the expected number of cx gates
        and provides the correct unitary.
        """
        qc = QuantumCircuit(num_ctrl_qubits + 1)
        qc.append(base_gate.control(num_ctrl_qubits), range(num_ctrl_qubits + 1))

        base_mat = base_gate.to_matrix()
        test_op = Operator(qc)
        cop_mat = _compute_control_matrix(base_mat, num_ctrl_qubits)
        self.assertTrue(matrix_equal(cop_mat, test_op.data))
        pm = generate_preset_pass_manager(
            optimization_level=0, basis_gates=["u", "cx"], seed_transpiler=12345
        )
        transpiled_circuit = pm.run(qc)
        cx_count = transpiled_circuit.count_ops()["cx"]
        expected = {1: 1, 2: 6, 3: 14, 4: 36}
        self.assertEqual(cx_count, expected[num_ctrl_qubits])

    @combine(
        num_ctrl_qubits=[1, 2, 3, 4],
        base_gate=[PhaseGate(0.123), SGate(), SdgGate(), TGate(), TdgGate(), SXGate(), SXdgGate()],
    )
    def test_small_mcp_gates_yield_cx_count(self, num_ctrl_qubits, base_gate):
        """Test that creating a MCPhase gate (and other locally equivalent multi-controlled gates)
        with small number of controls (with no ancillas) yields the expected number of cx gates
        and provides the correct unitary.
        """
        qc = QuantumCircuit(num_ctrl_qubits + 1)
        qc.append(base_gate.control(num_ctrl_qubits), range(num_ctrl_qubits + 1))
        base_mat = base_gate.to_matrix()
        test_op = Operator(qc)
        cop_mat = _compute_control_matrix(base_mat, num_ctrl_qubits)
        self.assertTrue(matrix_equal(cop_mat, test_op.data))

        # TODO: fix optimization_level=0 after updating CS and CSdg in the equivalence library
        pm = generate_preset_pass_manager(
            optimization_level=2, basis_gates=["u", "cx"], seed_transpiler=12345
        )
        transpiled_circuit = pm.run(qc)
        cx_count = transpiled_circuit.count_ops()["cx"]
        expected = {1: 2, 2: 6, 3: 20, 4: 44}
        self.assertEqual(cx_count, expected[num_ctrl_qubits])

    @combine(num_ctrl_qubits=[1, 2, 3], base_gate=[RXGate(0.789), RYGate(0.123), RZGate(0.456)])
    def test_small_mc_rotation_gates_yield_cx_count(self, num_ctrl_qubits, base_gate):
        """Test that creating a MCRX / MCRY / MCRZ gate
        with small number of controls (with no ancillas) yields the expected number of cx gates
        and provides the correct unitary.
        """
        qc = QuantumCircuit(num_ctrl_qubits + 1)
        qc.append(base_gate.control(num_ctrl_qubits), range(num_ctrl_qubits + 1))

        base_mat = base_gate.to_matrix()
        test_op = Operator(qc)
        cop_mat = _compute_control_matrix(base_mat, num_ctrl_qubits)
        self.assertTrue(matrix_equal(cop_mat, test_op.data))
        pm = generate_preset_pass_manager(
            optimization_level=0, basis_gates=["u", "cx"], seed_transpiler=12345
        )
        transpiled_circuit = pm.run(qc)
        cx_count = transpiled_circuit.count_ops()["cx"]
        if base_gate.name == "rz":
            expected = {1: 2, 2: 4, 3: 14}
        else:
            expected = {1: 2, 2: 8, 3: 20}
        self.assertEqual(cx_count, expected[num_ctrl_qubits])


if __name__ == "__main__":
    unittest.main()
