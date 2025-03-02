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

from qiskit.quantum_info import Operator
from qiskit.circuit.library import XGate, C3XGate
from qiskit.synthesis.multi_controlled import synth_c3x
from qiskit.circuit._utils import _compute_control_matrix

from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestMCXSynth(QiskitTestCase):
    """Test MCX synthesis methods."""

    @staticmethod
    def mcx_matrix(num_ctrl_qubits: int):
        """Return matrix for the MCX gate with the given number of control qubits."""
        base_mat = XGate().to_matrix()
        return _compute_control_matrix(base_mat, num_ctrl_qubits)

    def test_c3x(self):
        """Test the default synthesis method for C3XGate."""
        expected_mat = self.mcx_matrix(3)
        self.assertEqual(Operator(synth_c3x()), Operator(expected_mat))


if __name__ == "__main__":
    unittest.main()
