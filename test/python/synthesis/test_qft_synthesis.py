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

from qiskit.circuit.library import QFT
from qiskit.synthesis.qft import synth_qft_line
from qiskit.quantum_info import Operator
from qiskit.synthesis.linear.linear_circuits_utils import check_lnn_connectivity
from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestQFTLNN(QiskitTestCase):
    """Tests for QFT synthesis functions."""

    @combine(num_qubits=[2, 3, 4, 5, 6, 7, 8], do_swaps=[True, False])
    def test_qft_lnn(self, num_qubits, do_swaps):
        """Assert that the original and synthesized QFT circuits are the same."""
        qft_circ = QFT(num_qubits, do_swaps=do_swaps)
        qft_lnn = synth_qft_line(num_qubits, do_swaps=do_swaps)

        with self.subTest(msg="original and synthesized QFT circuits are not the same"):
            self.assertEqual(Operator(qft_circ), Operator(qft_lnn))

        # Check that the output circuit has LNN connectivity
        with self.subTest(msg="synthesized QFT circuit do not have LNN connectivity"):
            self.assertTrue(check_lnn_connectivity(qft_lnn))

    @combine(num_qubits=[5, 6, 7, 8], do_swaps=[True, False], approximation_degree=[2, 3])
    def test_qft_lnn_approximated(self, num_qubits, do_swaps, approximation_degree):
        """Assert that the original and synthesized QFT circuits are the same with approximation."""
        qft_circ = QFT(num_qubits, do_swaps=do_swaps, approximation_degree=approximation_degree)
        qft_lnn = synth_qft_line(
            num_qubits, do_swaps=do_swaps, approximation_degree=approximation_degree
        )

        with self.subTest(msg="original and synthesized QFT circuits are not the same"):
            self.assertEqual(Operator(qft_circ), Operator(qft_lnn))

        # Check that the output circuit has LNN connectivity
        with self.subTest(msg="synthesized QFT circuit do not have LNN connectivity"):
            self.assertTrue(check_lnn_connectivity(qft_lnn))


if __name__ == "__main__":
    unittest.main()
