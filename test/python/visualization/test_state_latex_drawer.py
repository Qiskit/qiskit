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

from qiskit.quantum_info import Statevector
from qiskit.visualization.state_visualization import state_drawer
from .visualization import QiskitVisualizationTestCase


class TestLatexStateDrawer(QiskitVisualizationTestCase):
    """Qiskit state and unitary latex drawer."""

    def test_state(self):
        """Test latex state vector drawer works with default settings."""

        sv = Statevector.from_label("+-rl")
        output = state_drawer(sv, "latex_source")
        expected_output = (
            r"\frac{1}{4} |0000\rangle- \frac{i}{4} |0001\rangle+\frac{i}{4} |0010\rangle"
            r"+\frac{1}{4} |0011\rangle- \frac{1}{4} |0100\rangle+\frac{i}{4} |0101\rangle"
            r" + \ldots +\frac{1}{4} |1011\rangle- \frac{1}{4} |1100\rangle"
            r"+\frac{i}{4} |1101\rangle- \frac{i}{4} |1110\rangle- \frac{1}{4} |1111\rangle"
        )
        self.assertEqual(output, expected_output)

    def test_state_max_size(self):
        """Test `max_size` parameter for latex ket notation."""

        sv = Statevector.from_label("+-rl")
        output = state_drawer(sv, "latex_source", max_size=4)
        expected_output = (
            r"\frac{1}{4} |0000\rangle- \frac{i}{4} |0001\rangle"
            r" + \ldots - \frac{1}{4} |1111\rangle"
        )
        self.assertEqual(output, expected_output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
