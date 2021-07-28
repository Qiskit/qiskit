# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Final perutation testing"""
import numpy as np
import unittest

from qiskit import QuantumCircuit, transpile
from qiskit.test import QiskitTestCase


class TestFinalPermutation(QiskitTestCase):
    """Tests for the final permutation"""

    def test_ghz_circuit_trivial(self):
        """Test 5Q GHZ with trivial layout"""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.measure_all()

        trans_qc = transpile(
            qc,
            basis_gates=["sx", "rz", "cx"],
            coupling_map=[[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]],
            initial_layout=[0, 1, 2, 3, 4],
            seed_transpiler=12345,
        )

        self.assertTrue(
            np.allclose(trans_qc.metadata["final_permutation"], np.array([2, 3, 0, 4, 1]))
        )

    def test_ghz_circuit_seed12(self):
        """Test 5Q GHZ with non-trivial layout"""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.measure_all()

        trans_qc = transpile(
            qc,
            basis_gates=["sx", "rz", "cx"],
            coupling_map=[[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]],
            seed_transpiler=12,
        )

        self.assertTrue(
            np.allclose(trans_qc.metadata["final_permutation"], np.array([2, 4, 0, 3, 1]))
        )

    def test_subcircuit(self):
        "Test that subcircuit permutations are correct"
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.measure_all()

        trans_qc = transpile(
            qc,
            basis_gates=["sx", "rz", "cx"],
            coupling_map=[[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]],
            initial_layout=[0, 2, 3, 1, 4],
            seed_transpiler=124,
        )
        perm1 = trans_qc.metadata["final_permutation"]

        qc2 = QuantumCircuit(3)
        qc2.h(0)
        qc2.cx(0, 1)
        qc2.cx(0, 2)
        qc2.measure_all()

        trans_qc2 = transpile(
            qc2,
            basis_gates=["sx", "rz", "cx"],
            coupling_map=[[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]],
            initial_layout=[0, 2, 3],
            seed_transpiler=124,
        )
        perm2 = trans_qc2.metadata["final_permutation"]

        self.assertTrue(np.allclose(perm1, perm2))


if __name__ == "__main__":
    unittest.main()
