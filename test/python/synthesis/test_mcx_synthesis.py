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

"""Test MCX synthesis."""

import unittest
import numpy as np
from ddt import ddt, data

from qiskit.quantum_info import Operator
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import XGate
from qiskit.synthesis.multi_controlled import synth_c3x, synth_mcx_n_dirty_i15, synth_mcx_noaux_v24
from qiskit.circuit._utils import _compute_control_matrix
from qiskit.quantum_info.operators.operator_utils import _equal_with_ancillas

from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestMCXSynth(QiskitTestCase):
    """Test MCX synthesis methods."""

    @staticmethod
    def mcx_matrix(num_ctrl_qubits: int):
        """Return matrix for the MCX gate with the given number of control qubits."""
        base_mat = XGate().to_matrix()
        return _compute_control_matrix(base_mat, num_ctrl_qubits)

    @staticmethod
    def check_mcx_synthesis(
        num_ctrl_qubits: int, synthesized_circuit: QuantumCircuit, clean_ancillas: bool
    ) -> bool:
        """Check correctness of a quantum circuit produced by an MCX synthesis algorithm, taking
        the additional ancilla qubits used into account.

        Args:
            num_ctrl_qubits: the number of control qubits for the MCX gate.
            synthesized_circuit: the quantum circuit synthesizing the MCX gate.
            clean_ancillas: True if the algorithm uses clean ancilla qubits.

        Returns:
            A Boolean indicating whether the synthesized circuit correctly implements the MCX gate.
        """
        original_op = Operator(TestMCXSynth.mcx_matrix(num_ctrl_qubits))
        synthesized_op = Operator(synthesized_circuit)

        num_qubits_original = original_op._op_shape._num_qargs_l
        num_qubits__synthesized = synthesized_circuit.num_qubits

        expected_op = Operator(
            np.kron(np.eye(2 ** (num_qubits__synthesized - num_qubits_original)), original_op)
        )
        if clean_ancillas:
            ancilla_qubits = list(range(num_qubits_original, num_qubits__synthesized))
        else:
            ancilla_qubits = []

        result = _equal_with_ancillas(
            synthesized_op,
            expected_op,
            ancilla_qubits,
        )
        return result

    def test_c3x(self):
        """Test synth_mcx_noaux_v24."""
        expected_mat = self.mcx_matrix(3)
        self.assertEqual(Operator(synth_c3x()), Operator(expected_mat))

    @data(1, 2, 3, 4, 5, 6, 7, 8)
    def test_mcx_noaux_v24(self, num_ctrl_qubits: int):
        """Test synth_mcx_noaux_v24."""
        synthesized_circuit = synth_mcx_noaux_v24(num_ctrl_qubits)
        self.assertTrue(
            self.check_mcx_synthesis(num_ctrl_qubits, synthesized_circuit, clean_ancillas=False)
        )

    @data(1, 2, 3, 4, 5, 6)
    def test_mcx_n_dirty_i15(self, num_ctrl_qubits: int):
        """Test synth_mcx_n_dirty_i15."""
        synthesized_circuit = synth_mcx_n_dirty_i15(num_ctrl_qubits)
        self.assertTrue(
            self.check_mcx_synthesis(num_ctrl_qubits, synthesized_circuit, clean_ancillas=False)
        )


if __name__ == "__main__":
    unittest.main()
