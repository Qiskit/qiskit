# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for visualization of latex state and unitary drawers"""

import unittest

from qiskit.quantum_info import Statevector
from qiskit.visualization.state_visualization import state_drawer
from qiskit.utils import optionals
from .visualization import QiskitVisualizationTestCase


class TestLatexStateDrawer(QiskitVisualizationTestCase):
    """Qiskit state and unitary latex drawer."""

    @unittest.skipUnless(optionals.HAS_SYMPY, "needs sympy")
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

    @unittest.skipUnless(optionals.HAS_SYMPY, "needs sympy")
    def test_state_max_size(self):
        """Test `max_size` parameter for latex ket notation."""

        sv = Statevector.from_label("+-rl")
        output = state_drawer(sv, "latex_source", max_size=4)
        expected_output = (
            r"\frac{1}{4} |0000\rangle- \frac{i}{4} |0001\rangle"
            r" + \ldots - \frac{1}{4} |1111\rangle"
        )
        self.assertEqual(output, expected_output)

    @unittest.skipUnless(optionals.HAS_SYMPY, "needs sympy")
    def test_state_ket_basis_z(self):
        """Test state_to_latex with z basis (default)."""
        from qiskit.visualization.state_visualization import state_to_latex
        
        sv = Statevector.from_label("0")
        latex_default = state_to_latex(sv)
        latex_z = state_to_latex(sv, ket_basis="z")
        # default should be same as explicit z basis
        self.assertEqual(latex_default, latex_z)
        self.assertIn("|0", latex_z)

    @unittest.skipUnless(optionals.HAS_SYMPY, "needs sympy")
    def test_state_ket_basis_x(self):
        """Test state_to_latex with x basis."""
        from qiskit.visualization.state_visualization import state_to_latex
        
        sv = Statevector.from_label("0")
        latex_x = state_to_latex(sv, ket_basis="x")
        # |0> in X-basis should show as superposition (|0>+|1>)/sqrt(2)
        self.assertIn("|0", latex_x)
        self.assertIn("|1", latex_x)
        # expecting \frac{\sqrt{2}}{2} |0\rangle+\frac{\sqrt{2}}{2} |1\rangle
        self.assertIn(r"\frac", latex_x)

    @unittest.skipUnless(optionals.HAS_SYMPY, "needs sympy")
    def test_state_ket_basis_y(self):
        """Test state_to_latex with y basis."""
        from qiskit.visualization.state_visualization import state_to_latex
        
        sv = Statevector.from_label("0")
        latex_y = state_to_latex(sv, ket_basis="y")
        # y-basis should have imaginary components
        # expecting \frac{\sqrt{2}}{2} |0\rangle- \frac{\sqrt{2} i}{2} |1\rangle
        self.assertIn("i", latex_y)
        self.assertIn("|0", latex_y)
        self.assertIn("|1", latex_y)

    @unittest.skipUnless(optionals.HAS_SYMPY, "needs sympy")
    def test_state_ket_basis_h(self):
        """Test state_to_latex with h basis (alias for x)."""
        from qiskit.visualization.state_visualization import state_to_latex
        
        sv = Statevector.from_label("0")
        latex_x = state_to_latex(sv, ket_basis="x")
        latex_h = state_to_latex(sv, ket_basis="h")
        # h and x bases should produce identical output
        self.assertEqual(latex_x, latex_h)

    @unittest.skipUnless(optionals.HAS_SYMPY, "needs sympy")
    def test_state_ket_basis_invalid(self):
        """Test state_to_latex with invalid basis raises error."""
        from qiskit.visualization.state_visualization import state_to_latex
        from qiskit.exceptions import VisualizationError
        
        sv = Statevector.from_label("0")
        with self.assertRaises(VisualizationError):
            state_to_latex(sv, ket_basis="invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
