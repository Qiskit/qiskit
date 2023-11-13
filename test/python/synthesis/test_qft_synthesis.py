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

"""Tests for QFT synthesis methods."""


import unittest
from test import combine
from ddt import ddt

from qiskit.test import QiskitTestCase
from qiskit.circuit.library import QFT
from qiskit.synthesis.qft import synth_qft_line
from qiskit.quantum_info import Operator


@ddt
class TestQFTLNN(QiskitTestCase):
    """Tests for QFT synthesis functions."""

    @combine(num_qubits=[4, 5, 6, 7])
    def test_qft_lnn(self, num_qubits):
        """Assert that the original and synthesized QFT circuits are the same."""
        qft_circ = QFT(num_qubits, do_swaps=True)
        qft_lnn = synth_qft_line(num_qubits)

        self.assertTrue(Operator(qft_circ).equiv(Operator(qft_lnn)))


if __name__ == "__main__":
    unittest.main()
