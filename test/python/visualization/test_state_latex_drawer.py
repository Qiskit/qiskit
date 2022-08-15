# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for visualization of latex state and unitary drawers"""

import unittest
import numpy as np

from qiskit.quantum_info import Statevector
from qiskit.visualization.state_visualization import state_drawer, state_to_latex, numbers_to_latex_terms
from .visualization import QiskitVisualizationTestCase


class TestLatexStateDrawer(QiskitVisualizationTestCase):
    """Qiskit state and unitary latex drawer."""

    def test_state(self):
        """Test latex state vector drawer works with default settings."""

        sv = Statevector.from_label("+-rl")
        output = state_drawer(sv, "latex_source")
        expected_output = (
            r"\frac{1}{4} |0000\rangle - \frac{i}{4} |0001\rangle +\frac{i}{4} |0010\rangle"
            r" +\frac{1}{4} |0011\rangle - \frac{1}{4} |0100\rangle +\frac{i}{4} |0101\rangle"
            r" + \ldots +\frac{1}{4} |1011\rangle - \frac{1}{4} |1100\rangle"
            r" +\frac{i}{4} |1101\rangle - \frac{i}{4} |1110\rangle - \frac{1}{4} |1111\rangle"
        )
        self.assertEqual(output, expected_output)

    def test_state_max_size(self):
        """Test `max_size` parameter for latex ket notation."""

        sv = Statevector.from_label("+-rl")
        output = state_drawer(sv, "latex_source", max_size=4)
        expected_output = (
            r"\frac{1}{4} |0000\rangle - \frac{i}{4} |0001\rangle"
            r" + \ldots - \frac{1}{4} |1111\rangle"
        )
        self.assertEqual(output, expected_output)

    def test_ket_prefix(self):
        """Test `prefix` parameter for latex ket notation."""

        sv = Statevector.from_label("+-rl")
        output = state_drawer(sv, "latex_source", prefix=r"|\psi\rangle=")
        expected_output = (
                "|\\psi\\rangle=\\frac{1}{4} |0000\\rangle - \\frac{i}{4} "
                "|0001\\rangle +\\frac{i}{4} |0010\\rangle +\\frac{1}{4} "
                "|0011\\rangle - \\frac{1}{4} |0100\\rangle +\\frac{i}{4} "
                "|0101\\rangle + \\ldots +\\frac{1}{4} |1011\\rangle - "
                "\\frac{1}{4} |1100\\rangle +\\frac{i}{4} |1101\\rangle - "
                "\\frac{i}{4} |1110\\rangle - \\frac{1}{4} |1111\\rangle"
        )
        self.assertEqual(output, expected_output)

    def test_state_to_latex_for_none(self):
        """
        Test for `\rangleNone` output in latex representation
        See https://github.com/Qiskit/qiskit-terra/issues/8169
        """
        sv = Statevector(
            [
                7.07106781e-01 - 8.65956056e-17j,
                -5.55111512e-17 - 8.65956056e-17j,
                7.85046229e-17 + 8.65956056e-17j,
                -7.07106781e-01 + 8.65956056e-17j,
                0.00000000e00 + 0.00000000e00j,
                -0.00000000e00 + 0.00000000e00j,
                -0.00000000e00 + 0.00000000e00j,
                0.00000000e00 - 0.00000000e00j,
            ],
            dims=(2, 2, 2),
        )
        latex_representation = state_to_latex(sv)
        self.assertEqual(
            latex_representation,
            "\\frac{\\sqrt{2}}{2} |000\\rangle - \\frac{\\sqrt{2}}{2} |011\\rangle",
        )

    def test_state_to_latex_for_large_statevector(self):
        """Test conversion of large dense state vector"""
        sv = Statevector(np.ones((2**15, 1)))
        latex_representation = state_to_latex(sv)
        self.assertEqual(
            latex_representation,
            "|000000000000000\\rangle + |000000000000001\\rangle + |000000000000010\\rangle +"
            " |000000000000011\\rangle + |000000000000100\\rangle + |000000000000101\\rangle +"
            " \\ldots + |111111111111011\\rangle + |111111111111100\\rangle +"
            " |111111111111101\\rangle + |111111111111110\\rangle + |111111111111111\\rangle",
        )

    def test_state_to_latex_for_large_sparse_statevector(self):
        """Test conversion of large sparse state vector"""
        sv = Statevector(np.eye(2**15, 1))
        latex_representation = state_to_latex(sv)
        self.assertEqual(latex_representation, "|000000000000000\\rangle")

    def test_number_to_latex_terms(self):
        """Test conversions of complex numbers to latex terms"""

        cases = [
            ([1 - 8e-17, -1+8e-17], ["", "-"]),
            ([1, -1], ["", "-"]),
            ([1j, 1j], ["i", "+i"]),
            ([-1, 1], ["-", "+"]),
            ([1j, 1], ["i", "+"]),
            ([-1, 1j], ["-", "+i"]),
            ([1e-16 + 1j], ["i"]),
            ([-1 + 1e-16 * 1j], ["-"]),
            ([-1, -1 - 1j], ["-", "+(-1 - i)"]),
            ([np.sqrt(2) / 2, np.sqrt(2) / 2], ["\\frac{\\sqrt{2}}{2}", "+\\frac{\\sqrt{2}}{2}"]),
            ([1 + np.sqrt(2)], ["(1 + \\sqrt{2})"]),
        ]
        for numbers, latex_terms in cases:
            terms = numbers_to_latex_terms(numbers, 15)
            self.assertListEqual(terms, latex_terms)



if __name__ == "__main__":
    unittest.main(verbosity=2)
