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

"""Tests for stabilizer state synthesis methods."""


import unittest
from test import combine
from ddt import ddt

import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.quantum_info.states import StabilizerState
from qiskit.quantum_info import random_clifford
from qiskit.synthesis.stabilizer import synth_stabilizer_layers

# from qiskit.synthesis.linear.linear_circuits_utils import check_lnn_connectivity


@ddt
class TestStabDecomposeLayers(QiskitTestCase):
    """Tests for stabilizer state decomposition functions."""

    @combine(num_qubits=[4, 5, 6, 7])
    def test_decompose_stab(self, num_qubits):
        """Create layer decomposition for a Clifford U, and check that it
        results in an equivalent Clifford."""
        rng = np.random.default_rng(1234)
        samples = 10
        for _ in range(samples):
            cliff = random_clifford(num_qubits, seed=rng)
            stab = StabilizerState(cliff)
            circ = synth_stabilizer_layers(stab)
            stab_target = StabilizerState(circ)
            # Verify that the two stabilizers generate the same state
            self.assertTrue(stab.equiv(stab_target))
            # Verify that the two stabilizers produce the same probabilities
            self.assertEqual(stab.probabilities_dict(), stab_target.probabilities_dict())
            # Verify the layered structure
            self.assertEqual(circ.data[0].operation.name, "H2")
            self.assertEqual(circ.data[1].operation.name, "S1")
            self.assertEqual(circ.data[2].operation.name, "CZ")
            self.assertEqual(circ.data[3].operation.name, "H1")
            self.assertEqual(circ.data[4].operation.name, "Pauli")


if __name__ == "__main__":
    unittest.main()
