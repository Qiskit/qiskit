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
    U1Gate,
    U2Gate,
    U3Gate,
    CZGate,
)
from qiskit.synthesis.multi_controlled import (
    synth_mcx_n_dirty_i15,
    synth_mcx_n_clean_m15,
    synth_mcx_1_clean_b95,
    synth_mcx_1_clean_kg24,
    synth_mcx_1_dirty_kg24,
    synth_mcx_2_clean_kg24,
    synth_mcx_2_dirty_kg24,
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
class TestMCSynthesisCorrectness(QiskitTestCase):
    """Test correctness of synthesis methods for multi-controlled gates."""

    @staticmethod
    def mc_matrix(base_gate: Gate, num_ctrl_qubits: int):
        """Return matrix for the MC gate with the given base gate and the number of control qubits."""
        base_mat = base_gate.to_matrix()
        return _compute_control_matrix(base_mat, num_ctrl_qubits)

    def assertSynthesisCorrect(
        self,
        base_gate: Gate,
        num_ctrl_qubits: int,
        synthesized_circuit: QuantumCircuit,
        clean_ancillas: bool,
    ):
        """Check correctness of a quantum circuit produced by a synthesis algorithm for multi-controlled
        gates, taking the additional ancilla qubits into account.

        This check is based on comparing the synthesized and the expected matrices and thus only
        works for synthesized circuits with up to about 10 qubits.

        Args:
            base_gate: the base gate of the MC gate.
            num_ctrl_qubits: the number of control qubits in the MC gate.
            synthesized_circuit: the quantum circuit synthesizing the MC gate.
            clean_ancillas: True if the algorithm uses clean ancilla qubits.

        Note: currently we do not have any MC synthesis algorithms that use both clean and dirty
        ancilla qubits. When we do, we will need to extend this function.
        """
        original_op = Operator(self.mc_matrix(base_gate, num_ctrl_qubits))
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
        self.assertSynthesisCorrect(
            XGate(), num_ctrl_qubits, synthesized_circuit, clean_ancillas=False
        )

    @data(3, 4, 5, 6)
    def test_mcx_n_clean_m15(self, num_ctrl_qubits: int):
        """Test synth_mcx_n_clean_m15 by comparing synthesized and expected matrices."""
        # Note: the method requires at least 3 control qubits
        synthesized_circuit = synth_mcx_n_clean_m15(num_ctrl_qubits)
        self.assertSynthesisCorrect(
            XGate(), num_ctrl_qubits, synthesized_circuit, clean_ancillas=True
        )

    @data(3, 4, 5, 6, 7, 8)
    def test_mcx_1_clean_b95(self, num_ctrl_qubits: int):
        """Test synth_mcx_1_clean_b95 by comparing synthesized and expected matrices."""
        # Note: the method requires at least 3 control qubits
        synthesized_circuit = synth_mcx_1_clean_b95(num_ctrl_qubits)
        self.assertSynthesisCorrect(
            XGate(), num_ctrl_qubits, synthesized_circuit, clean_ancillas=True
        )

    @data(3, 4, 5, 6, 7, 8)
    def test_mcx_1_clean_kg24(self, num_ctrl_qubits: int):
        """Test synth_mcx_1_clean_kg24 by comparing synthesized and expected matrices."""
        # Note: the method requires at least 3 control qubits
        synthesized_circuit = synth_mcx_1_clean_kg24(num_ctrl_qubits)
        self.assertSynthesisCorrect(
            XGate(), num_ctrl_qubits, synthesized_circuit, clean_ancillas=True
        )

    @data(3, 4, 5, 6, 7, 8)
    def test_mcx_1_dirty_kg24(self, num_ctrl_qubits: int):
        """Test synth_mcx_1_dirty_kg24 by comparing synthesized and expected matrices."""
        # Note: the method requires at least 3 control qubits
        synthesized_circuit = synth_mcx_1_dirty_kg24(num_ctrl_qubits)
        self.assertSynthesisCorrect(
            XGate(), num_ctrl_qubits, synthesized_circuit, clean_ancillas=False
        )

    @data(3, 4, 5, 6, 7, 8)
    def test_mcx_2_clean_kg24(self, num_ctrl_qubits: int):
        """Test synth_mcx_2_clean_kg24 by comparing synthesized and expected matrices."""
        # Note: the method requires at least 3 control qubits
        synthesized_circuit = synth_mcx_2_clean_kg24(num_ctrl_qubits)
        self.assertSynthesisCorrect(
            XGate(), num_ctrl_qubits, synthesized_circuit, clean_ancillas=True
        )

    @data(3, 4, 5, 6, 7, 8)
    def test_mcx_2_dirty_kg24(self, num_ctrl_qubits: int):
        """Test synth_mcx_2_dirty_kg24 by comparing synthesized and expected matrices."""
        # Note: the method requires at least 3 control qubits
        synthesized_circuit = synth_mcx_2_dirty_kg24(num_ctrl_qubits)
        self.assertSynthesisCorrect(
            XGate(), num_ctrl_qubits, synthesized_circuit, clean_ancillas=False
        )

    @data(3, 4, 5, 6, 7, 8)
    def test_mcx_gray_code(self, num_ctrl_qubits: int):
        """Test synth_mcx_gray_code by comparing synthesized and expected matrices."""
        # Note: the method requires at least 3 control qubits
        synthesized_circuit = synth_mcx_gray_code(num_ctrl_qubits)
        self.assertSynthesisCorrect(
            XGate(), num_ctrl_qubits, synthesized_circuit, clean_ancillas=False
        )

    @data(1, 2, 3, 4, 5, 6, 7, 8)
    def test_mcx_noaux_v24(self, num_ctrl_qubits: int):
        """Test synth_mcx_noaux_v24 by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_mcx_noaux_v24(num_ctrl_qubits)
        self.assertSynthesisCorrect(
            XGate(), num_ctrl_qubits, synthesized_circuit, clean_ancillas=False
        )

    def test_c3x(self):
        """Test synth_c3x by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_c3x()
        self.assertSynthesisCorrect(XGate(), 3, synthesized_circuit, clean_ancillas=False)

    def test_c4x(self):
        """Test synth_c4x by comparing synthesized and expected matrices."""
        synthesized_circuit = synth_c4x()
        self.assertSynthesisCorrect(XGate(), 4, synthesized_circuit, clean_ancillas=False)

    @combine(
        num_ctrl_qubits=[1, 2, 3, 4, 5, 6, 7],
        base_gate=[
            XGate(),
            YGate(),
            ZGate(),
            HGate(),
            PhaseGate(0.123),
            SGate(),
            SdgGate(),
            TGate(),
            TdgGate(),
            SXGate(),
            SXdgGate(),
            RXGate(0.789),
            RYGate(0.123),
            RZGate(0.456),
            UGate(0.1, 0.2, 0.3),
            U1Gate(0.1),
            U2Gate(0.1, 0.2),
            U3Gate(0.1, 0.2, 0.3),
        ],
        annotated=[False, True],
    )
    def test_create_mc_gates(self, num_ctrl_qubits, base_gate, annotated):
        """Test that creating various multi-controlled gates with small number of controls
        and no ancillas yields correct unitaries.
        """
        qc = QuantumCircuit(num_ctrl_qubits + 1)
        qc.append(
            base_gate.control(num_ctrl_qubits, annotated=annotated), range(num_ctrl_qubits + 1)
        )
        test_op = Operator(qc).data
        cop_mat = self.mc_matrix(base_gate, num_ctrl_qubits)
        self.assertTrue(matrix_equal(cop_mat, test_op))


@ddt
class TestMCSynthesisCounts(QiskitTestCase):
    """Test gate counts produced by multi-controlled synthesis methods."""

    def setUp(self):
        super().setUp()
        self.pm = generate_preset_pass_manager(
            optimization_level=0, basis_gates=["u", "cx"], seed_transpiler=12345
        )

    @data(5, 10, 15)
    def test_mcx_n_dirty_i15_cx_count(self, num_ctrl_qubits: int):
        """Test synth_mcx_n_dirty_i15 bound on CX count."""
        synthesized_circuit = synth_mcx_n_dirty_i15(num_ctrl_qubits)
        transpiled_circuit = self.pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The bound from the documentation of synth_mcx_n_dirty_i15
        self.assertLessEqual(cx_count, 8 * num_ctrl_qubits - 6)

    @data(5, 10, 15)
    def test_mcx_n_clean_m15_cx_count(self, num_ctrl_qubits: int):
        """Test synth_mcx_n_clean_m15 bound on CX count."""
        synthesized_circuit = synth_mcx_n_clean_m15(num_ctrl_qubits)
        transpiled_circuit = self.pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The bound from the documentation of synth_mcx_n_clean_m15
        self.assertLessEqual(cx_count, 6 * num_ctrl_qubits - 6)

    @data(5, 10, 15)
    def test_mcx_1_clean_b95_cx_count(self, num_ctrl_qubits: int):
        """Test synth_mcx_1_clean_b95 bound on CX count."""
        synthesized_circuit = synth_mcx_1_clean_b95(num_ctrl_qubits)
        transpiled_circuit = self.pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The bound from the documentation of synth_mcx_1_clean_b95
        self.assertLessEqual(cx_count, 16 * num_ctrl_qubits - 24)

    @data(3, 5, 10, 15)
    def test_mcx_1_clean_kg24_cx_count(self, num_ctrl_qubits: int):
        """Test synth_mcx_1_clean_kg24 bound on CX count."""
        synthesized_circuit = synth_mcx_1_clean_kg24(num_ctrl_qubits)
        transpiled_circuit = self.pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # Based on the bound from the Sec 5.1 of arXiv:2407.17966, assuming Toffoli decomposition
        # requires 6 CX gates.
        self.assertLessEqual(cx_count, 12 * num_ctrl_qubits - 18)

    @data(3, 5, 10, 15)
    def test_mcx_1_dirty_kg24_cx_count(self, num_ctrl_qubits: int):
        """Test synth_mcx_1_dirty_kg24 bound on CX count."""
        synthesized_circuit = synth_mcx_1_dirty_kg24(num_ctrl_qubits)
        transpiled_circuit = self.pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        ## Based on the bound from the Sec 5.3 of arXiv:2407.17966, assuming Toffoli decomposition
        # requires 6 CX gates.
        self.assertLessEqual(cx_count, 24 * num_ctrl_qubits - 48)

    @data(3, 5, 10, 15)
    def test_mcx_2_clean_kg24_cx_count(self, num_ctrl_qubits: int):
        """Test synth_mcx_2_clean_kg24 bound on CX count."""
        synthesized_circuit = synth_mcx_2_clean_kg24(num_ctrl_qubits)
        transpiled_circuit = self.pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # Based on the bound from the Sec 5.2 of arXiv:2407.17966, assuming Toffoli decomposition
        # requires 6 CX gates.
        self.assertLessEqual(cx_count, 12 * num_ctrl_qubits - 18)

    @data(3, 5, 10, 15)
    def test_mcx_2_dirty_kg24_cx_count(self, num_ctrl_qubits: int):
        """Test synth_mcx_2_dirty_kg24 bound on CX count."""
        synthesized_circuit = synth_mcx_2_dirty_kg24(num_ctrl_qubits)
        transpiled_circuit = self.pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # Based on the bound from the Sec 5.4 of arXiv:2407.17966, assuming Toffoli decomposition
        # requires 6 CX gates.
        self.assertLessEqual(cx_count, 24 * num_ctrl_qubits - 48)

    def test_c3x_cx_count(self):
        """Test synth_c3x bound on CX count."""
        synthesized_circuit = synth_c3x()
        transpiled_circuit = self.pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The bound from the default construction for C3X
        self.assertLessEqual(cx_count, 14)

    def test_c4x_cx_count(self):
        """Test synth_c4x bound on CX count."""
        synthesized_circuit = synth_c4x()
        transpiled_circuit = self.pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The bound from the default constuction for C4X
        self.assertLessEqual(cx_count, 36)

    @combine(
        num_ctrl_qubits=[5, 10, 15],
        base_gate=[RXGate(0.123), RYGate(0.456), RZGate(0.789)],
        annotated=[False, True],
    )
    def test_mc_rotation_gates_cx_count(
        self, num_ctrl_qubits: int, base_gate: Gate, annotated: bool
    ):
        """Test bounds on the number of CX gates for mcrx / mcry / mcrz."""
        qc = QuantumCircuit(num_ctrl_qubits + 1)
        qc.append(
            base_gate.control(num_ctrl_qubits, annotated=annotated), range(num_ctrl_qubits + 1)
        )
        transpiled_circuit = self.pm.run(qc)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The synthesis of mcrx/mcry/mcrz gates uses _mcsu2_real_diagonal.
        # The bounds are given in arXiv:2302.06377, Theorem 3.
        # In practice, we actually get better bounds for small values of num_ctrl_qubits.
        expected_cx_count = 16 * (num_ctrl_qubits + 1) - 40
        self.assertLessEqual(cx_count, expected_cx_count)

    @data(5, 10, 15)
    def test_mcx_noaux_v24_cx_count(self, num_ctrl_qubits: int):
        """Test synth_mcx_noaux_v24 bound on CX count."""
        synthesized_circuit = synth_mcx_noaux_v24(num_ctrl_qubits)
        transpiled_circuit = self.pm.run(synthesized_circuit)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The algorithm synth_mcx_noaux_v24 is based on the synthesis of MCPhase,
        # which is defined using a sequence of MCRZ gates:
        # MCPhase(n) is defined using one MCRZ(1), one MCRZ(2), ..., one MCRZ(n).
        # The bound below follows using the bound of 16*(k+1)-40 for MCRZ(k) and summing
        # the resulting arithmetic progression:
        #   sum_{k=1}^n (16*(k+1)-40) = sum_{k=1}^n (16*k - 24) =
        #     16*n*(n+1)/2 - 24*n = 8n^2 - 16*n.
        self.assertLessEqual(cx_count, 8 * num_ctrl_qubits**2 - 16 * num_ctrl_qubits)

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
        annotated=[False, True],
    )
    def test_mcx_equiv_noaux_cx_count(self, num_ctrl_qubits: int, base_gate: Gate, annotated: bool):
        """Test bounds on the number of CX-gates when synthesizing multi-controlled gates
        which are locally equivalent to MCX.
        """
        qc = QuantumCircuit(num_ctrl_qubits + 1)
        qc.append(
            base_gate.control(num_ctrl_qubits, annotated=annotated), range(num_ctrl_qubits + 1)
        )
        transpiled_circuit = self.pm.run(qc)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The bounds should be the same as for synth_mcx_noaux_v24
        self.assertLessEqual(cx_count, 8 * num_ctrl_qubits**2 - 16 * num_ctrl_qubits)

    @combine(
        num_ctrl_qubits=[5, 10, 15],
        annotated=[False, True],
    )
    def test_mcu_noaux_cx_count(self, num_ctrl_qubits: int, annotated: bool):
        """Test bounds on the number of CX-gates when synthesizing multi-controlled single-qubit
        unitary gates.
        """
        base_gate = UGate(0.123, 0.456, 0.789)
        qc = QuantumCircuit(num_ctrl_qubits + 1)
        qc.append(
            base_gate.control(num_ctrl_qubits, annotated=annotated), range(num_ctrl_qubits + 1)
        )
        transpiled_circuit = self.pm.run(qc)
        cx_count = transpiled_circuit.count_ops()["cx"]
        # The synthesis of MCX(n) uses two MCRZ(n), one MCRY(n), and one MCPhase(n-1).
        # Thus the number of CX-gate should be upper-bounded by
        # 3*(16 * (n + 1) - 40) + (8 * (n-1)^2 - 16 * (n-1))
        self.assertLessEqual(cx_count, 8 * num_ctrl_qubits**2 + 16 * num_ctrl_qubits - 96)

    @combine(
        num_ctrl_qubits=[1, 2, 3, 4, 5, 6, 7, 8],
        base_gate=[
            XGate(),
            YGate(),
            ZGate(),
            HGate(),
            PhaseGate(0.123),
            SGate(),
            SdgGate(),
            TGate(),
            TdgGate(),
            SXGate(),
            SXdgGate(),
            RXGate(0.789),
            RYGate(0.123),
            RZGate(0.456),
            UGate(0.1, 0.2, 0.3),
            U1Gate(0.1),
            U2Gate(0.1, 0.2),
            U3Gate(0.1, 0.2, 0.3),
            CZGate(),
        ],
        annotated=[False, True],
    )
    def test_small_mc_gates_cx_count(self, num_ctrl_qubits: int, base_gate: Gate, annotated: bool):
        """Test that transpiling various multi-controlled gates with small number of controls and no
        ancillas yields the expected number of CX gates.

        This test prevents making changes to the synthesis algorithms that would deteriorate the
        quality of the synthesized circuits.
        """
        qc = QuantumCircuit(num_ctrl_qubits + base_gate.num_qubits)
        qc.append(base_gate.control(num_ctrl_qubits, annotated=annotated), qc.qubits)
        transpiled_circuit = self.pm.run(qc)
        cx_count = transpiled_circuit.count_ops()["cx"]

        if isinstance(base_gate, (XGate, YGate, ZGate, HGate)):
            # MCX gate and other locally equivalent multi-controlled gates
            expected = {1: 1, 2: 6, 3: 14, 4: 36, 5: 84, 6: 140, 7: 220, 8: 324}
        elif isinstance(
            base_gate, (PhaseGate, SGate, SdgGate, TGate, TdgGate, SXGate, SXdgGate, U1Gate)
        ):
            # MCPhase gate and other locally equivalent multi-controlled gates
            expected = {1: 2, 2: 6, 3: 20, 4: 44, 5: 84, 6: 140, 7: 220, 8: 324}
        elif isinstance(base_gate, RZGate):
            expected = {1: 2, 2: 4, 3: 14, 4: 24, 5: 40, 6: 56, 7: 80, 8: 104}
        elif isinstance(base_gate, (RXGate, RYGate)):
            expected = {1: 2, 2: 8, 3: 20, 4: 24, 5: 40, 6: 56, 7: 80, 8: 104}
        elif isinstance(base_gate, (UGate, U2Gate, U3Gate)):
            expected = {1: 2, 2: 22, 3: 54, 4: 92, 5: 164, 6: 252, 7: 380, 8: 532}
        elif isinstance(base_gate, CZGate):
            expected = {1: 6, 2: 14, 3: 36, 4: 84, 5: 140, 6: 220, 7: 324, 8: 444}
        else:
            raise NotImplementedError

        self.assertLessEqual(cx_count, expected[num_ctrl_qubits])


if __name__ == "__main__":
    unittest.main()
