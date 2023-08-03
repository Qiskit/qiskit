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

"""Test CZ circuits synthesis functions."""

import unittest
from test import combine
import numpy as np
from ddt import ddt
from qiskit import QuantumCircuit
from qiskit.circuit.library import Permutation
from qiskit.synthesis.linear_phase import synth_cz_depth_line_mr
from qiskit.synthesis.linear.linear_circuits_utils import check_lnn_connectivity
from qiskit.quantum_info import Clifford
from qiskit.test import QiskitTestCase


@ddt
class TestCZSynth(QiskitTestCase):
    """Test the linear reversible circuit synthesis functions."""

    @combine(num_qubits=[3, 4, 5, 6, 7])
    def test_cz_synth_lnn(self, num_qubits):
        """Test the CZ synthesis code for linear nearest neighbour connectivity."""
        seed = 1234
        rng = np.random.default_rng(seed)
        num_gates = 10
        num_trials = 5
        for _ in range(num_trials):
            mat = np.zeros((num_qubits, num_qubits))
            qctest = QuantumCircuit(num_qubits)

            # Generate a random CZ circuit
            for _ in range(num_gates):
                i = rng.integers(num_qubits)
                j = rng.integers(num_qubits)
                if i != j:
                    qctest.cz(i, j)
                    if j > i:
                        mat[i][j] = (mat[i][j] + 1) % 2
                    else:
                        mat[j][i] = (mat[j][i] + 1) % 2

            qc = synth_cz_depth_line_mr(mat)
            # Check that the output circuit 2-qubit depth equals to 2*n+2
            depth2q = qc.depth(filter_function=lambda x: x.operation.num_qubits == 2)
            self.assertTrue(depth2q == 2 * num_qubits + 2)
            # Check that the output circuit has LNN connectivity
            self.assertTrue(check_lnn_connectivity(qc))
            # Assert that we get the same element, up to reverse order of qubits
            perm = Permutation(num_qubits=num_qubits, pattern=range(num_qubits)[::-1])
            qctest = qctest.compose(perm)
            self.assertEqual(Clifford(qc), Clifford(qctest))


if __name__ == "__main__":
    unittest.main()
