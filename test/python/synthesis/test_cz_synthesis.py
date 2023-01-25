# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test linear reversible circuits synthesis functions."""

import unittest

import random
from test import combine
import numpy as np
from ddt import ddt
from qiskit import QuantumCircuit
from qiskit.circuit.library import Permutation
from qiskit.synthesis.linear import synth_cz_depth_line_mr
from qiskit.quantum_info import Clifford
from qiskit.test import QiskitTestCase


@ddt
class TestCZSynth(QiskitTestCase):
    """Test the linear reversible circuit synthesis functions."""

    @combine(num_qubits=[4, 5, 6, 7])
    def test_cz_synth_lnn(self, num_qubits):
        """Test the CZ synthesis code."""
        mat = np.zeros((num_qubits, num_qubits))
        qctest = QuantumCircuit(num_qubits)
        samples = 10
        for _ in range(samples):
            # Generate a random CZ circuit
            i = random.randint(0, num_qubits - 1)
            j = random.randint(0, num_qubits - 1)
            if i != j:
                qctest.cz(i, j)
                if j > i:
                    mat[i][j] = (mat[i][j] + 1) % 2
                else:
                    mat[j][i] = (mat[j][i] + 1) % 2
        qc = synth_cz_depth_line_mr(mat)
        perm = Permutation(num_qubits=num_qubits, pattern=range(num_qubits)[::-1])
        qctest = qctest.compose(perm)
        self.assertEqual(Clifford(qc), Clifford(qctest))


if __name__ == "__main__":
    unittest.main()
