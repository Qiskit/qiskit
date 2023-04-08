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

"""Test -CZ-CX- joint synthesis function."""

import unittest
from test import combine
import numpy as np
from ddt import ddt
from qiskit import QuantumCircuit
from qiskit.synthesis.linear.linear_depth_lnn import synth_cnot_depth_line_kms
from qiskit.synthesis.linear_phase.cx_cz_depth_lnn import synth_cx_cz_line_my

from qiskit.quantum_info import Clifford

from qiskit.synthesis.linear import (
    synth_cnot_count_full_pmh,
    synth_cnot_depth_line_kms,
    random_invertible_binary_matrix,
    check_invertible_binary_matrix,
    calc_inverse_matrix,
)

from qiskit.synthesis.linear.linear_circuits_utils import (
    transpose_cx_circ, 
    optimize_cx_4_options,
    check_lnn_connectivity)

from qiskit.test import QiskitTestCase

@ddt
class TestCXCZSynth(QiskitTestCase):
    """Test the linear reversible circuit synthesis functions."""

    @combine(num_qubits=[3, 4, 5, 6, 7, 8, 9, 10])
    def test_cx_cz_synth_lnn(self, num_qubits):
        """Test the CXCZ synthesis code for linear nearest neighbour connectivity."""
        seed = 1234
        rng = np.random.default_rng(seed)
        num_gates = 10
        num_trials = 8

        for _ in range(num_trials):


            # Generate a random CZ circuit
            mat_z = np.zeros((num_qubits, num_qubits))
            cir_z = QuantumCircuit(num_qubits)
            for _ in range(num_gates):
                i = rng.integers(num_qubits)
                j = rng.integers(num_qubits)
                if i != j:
                    cir_z.cz(i, j)
                    if j > i:
                        mat_z[i][j] = (mat_z[i][j] + 1) % 2
                    else:
                        mat_z[j][i] = (mat_z[j][i] + 1) % 2

            # Generate a random CX circuit
            mat_x = random_invertible_binary_matrix(num_qubits, seed=rng)
            mat_x = np.array(mat_x, dtype=bool)
            cir_x = synth_cnot_depth_line_kms(mat_x)


            # Joint Synthesis

            cirZX_test = QuantumCircuit.compose(cir_z, cir_x)

            cirZX = synth_cx_cz_line_my(mat_x,mat_z)

            # Check that the output circuit 2-qubit depth is at most 5n

            depth2q = cirZX.depth(filter_function=lambda x: x.operation.num_qubits == 2)
            self.assertTrue(depth2q <= 5 * num_qubits )

            # Check that the output circuit has LNN connectivity
            self.assertTrue(check_lnn_connectivity(cirZX))

            # Assert that we get the same elements as other methods
            self.assertEqual(Clifford(cirZX), Clifford(cirZX_test))


if __name__ == "__main__":
    unittest.main()
