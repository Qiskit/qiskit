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

# pylint: disable=missing-docstring

import unittest
import os
from unittest.mock import patch

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.test import QiskitTestCase
from qiskit.utils import optionals
from qiskit import visualization
from qiskit.visualization.circuit import text
from qiskit.visualization.exceptions import VisualizationError

if optionals.HAS_MATPLOTLIB:
    from matplotlib import figure
if optionals.HAS_PIL:
    from PIL import Image


_latex_drawer_condition = unittest.skipUnless(
    all(
        (
            optionals.HAS_PYLATEX,
            optionals.HAS_PIL,
            optionals.HAS_PDFLATEX,
            optionals.HAS_PDFTOCAIRO,
        )
    ),
    "Skipped because not all of PIL, pylatex, pdflatex and pdftocairo are available",
)


class TestCircuitDrawer(QiskitTestCase):
    def test_default_output(self):
        with patch("qiskit.user_config.get_config", return_value={}):
            circuit = QuantumCircuit()
            out = visualization.circuit_drawer(circuit)
            self.assertIsInstance(out, text.TextDrawing)

    @unittest.skipUnless(optionals.HAS_MATPLOTLIB, "Skipped because matplotlib is not available")
    def test_user_config_default_output(self):
        with patch("qiskit.user_config.get_config", return_value={"circuit_drawer": "mpl"}):
            circuit = QuantumCircuit()
            out = visualization.circuit_drawer(circuit)
            self.assertIsInstance(out, figure.Figure)

    def test_default_output_with_user_config_not_set(self):
        with patch("qiskit.user_config.get_config", return_value={"other_option": True}):
            circuit = QuantumCircuit()
            out = visualization.circuit_drawer(circuit)
            self.assertIsInstance(out, text.TextDrawing)

    @unittest.skipUnless(optionals.HAS_MATPLOTLIB, "Skipped because matplotlib is not available")
    def test_kwarg_priority_over_user_config_default_output(self):
        with patch("qiskit.user_config.get_config", return_value={"circuit_drawer": "latex"}):
            circuit = QuantumCircuit()
            out = visualization.circuit_drawer(circuit, output="mpl")
            self.assertIsInstance(out, figure.Figure)

    @unittest.skipUnless(optionals.HAS_MATPLOTLIB, "Skipped because matplotlib is not available")
    def test_default_backend_auto_output_with_mpl(self):
        with patch("qiskit.user_config.get_config", return_value={"circuit_drawer": "auto"}):
            circuit = QuantumCircuit()
            out = visualization.circuit_drawer(circuit)
            self.assertIsInstance(out, figure.Figure)

    def test_default_backend_auto_output_without_mpl(self):
        with patch("qiskit.user_config.get_config", return_value={"circuit_drawer": "auto"}):
            with optionals.HAS_MATPLOTLIB.disable_locally():
                circuit = QuantumCircuit()
                out = visualization.circuit_drawer(circuit)
                self.assertIsInstance(out, text.TextDrawing)

    @_latex_drawer_condition
    def test_latex_unsupported_image_format_error_message(self):
        with patch("qiskit.user_config.get_config", return_value={"circuit_drawer": "latex"}):
            circuit = QuantumCircuit()
            with self.assertRaises(VisualizationError, msg="Pillow could not write the image file"):
                visualization.circuit_drawer(circuit, filename="file.spooky")

    @_latex_drawer_condition
    def test_latex_output_file_correct_format(self):
        with patch("qiskit.user_config.get_config", return_value={"circuit_drawer": "latex"}):
            circuit = QuantumCircuit()
            filename = "file.gif"
            visualization.circuit_drawer(circuit, filename=filename)
            with Image.open(filename) as im:
                if filename.endswith("jpg"):
                    self.assertIn(im.format.lower(), "jpeg")
                else:
                    self.assertIn(im.format.lower(), filename.split(".")[-1])
            os.remove(filename)

    def test_wire_order(self):
        """Test wire_order
        See: https://github.com/Qiskit/qiskit-terra/pull/9893"""
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(4, "c")
        cr2 = ClassicalRegister(2, "ca")
        circuit = QuantumCircuit(qr, cr, cr2)
        circuit.h(0)
        circuit.h(3)
        circuit.x(1)
        circuit.x(3).c_if(cr, 10)

        expected = "\n".join(
            [
                "                  ",
                " q_2: ────────────",
                "      ┌───┐ ┌───┐ ",
                " q_3: ┤ H ├─┤ X ├─",
                "      ├───┤ └─╥─┘ ",
                " q_0: ┤ H ├───╫───",
                "      ├───┤   ║   ",
                " q_1: ┤ X ├───╫───",
                "      └───┘┌──╨──┐",
                " c: 4/═════╡ 0xa ╞",
                "           └─────┘",
                "ca: 2/════════════",
                "                  ",
            ]
        )
        result = visualization.circuit_drawer(circuit, wire_order=[2, 3, 0, 1])
        self.assertEqual(result.__str__(), expected)

    def test_wire_order_cregbundle(self):
        """Test wire_order with cregbundle=True
        See: https://github.com/Qiskit/qiskit-terra/pull/9893"""
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(4, "c")
        cr2 = ClassicalRegister(2, "ca")
        circuit = QuantumCircuit(qr, cr, cr2)
        circuit.h(0)
        circuit.h(3)
        circuit.x(1)
        circuit.x(3).c_if(cr, 10)

        expected = "\n".join(
            [
                "                  ",
                " q_2: ────────────",
                "      ┌───┐ ┌───┐ ",
                " q_3: ┤ H ├─┤ X ├─",
                "      ├───┤ └─╥─┘ ",
                " q_0: ┤ H ├───╫───",
                "      ├───┤   ║   ",
                " q_1: ┤ X ├───╫───",
                "      └───┘┌──╨──┐",
                " c: 4/═════╡ 0xa ╞",
                "           └─────┘",
                "ca: 2/════════════",
                "                  ",
            ]
        )
        result = visualization.circuit_drawer(circuit, wire_order=[2, 3, 0, 1], cregbundle=True)
        self.assertEqual(result.__str__(), expected)

    def test_wire_order_raises(self):
        """Verify we raise if using wire order incorrectly."""

        circuit = QuantumCircuit(3, 3)
        circuit.x(1)

        with self.assertRaisesRegex(VisualizationError, "should not have repeated elements"):
            visualization.circuit_drawer(circuit, wire_order=[2, 1, 0, 3, 1, 5])

        with self.assertRaisesRegex(VisualizationError, "cannot be set when the reverse_bits"):
            visualization.circuit_drawer(circuit, wire_order=[0, 1, 2, 5, 4, 3], reverse_bits=True)

        with self.assertWarnsRegex(RuntimeWarning, "cregbundle set"):
            visualization.circuit_drawer(circuit, cregbundle=True, wire_order=[0, 1, 2, 5, 4, 3])

    def test_reverse_bits(self):
        """Test reverse_bits should not raise warnings when no classical qubits:
        See: https://github.com/Qiskit/qiskit-terra/pull/8689"""
        circuit = QuantumCircuit(3)
        circuit.x(1)
        expected = "\n".join(
            [
                "          ",
                "q_2: ─────",
                "     ┌───┐",
                "q_1: ┤ X ├",
                "     └───┘",
                "q_0: ─────",
                "          ",
            ]
        )
        result = visualization.circuit_drawer(circuit, output="text", reverse_bits=True)
        self.assertEqual(result.__str__(), expected)

    def test_no_explict_cregbundle(self):
        """Test no explicit cregbundle should not raise warnings about being disabled
        See: https://github.com/Qiskit/qiskit-terra/issues/8690"""
        inner = QuantumCircuit(1, 1, name="inner")
        inner.measure(0, 0)
        circuit = QuantumCircuit(2, 2)
        circuit.append(inner, [0], [0])
        expected = "\n".join(
            [
                "     ┌────────┐",
                "q_0: ┤0       ├",
                "     │        │",
                "q_1: ┤  inner ├",
                "     │        │",
                "c_0: ╡0       ╞",
                "     └────────┘",
                "c_1: ══════════",
                "               ",
            ]
        )
        result = circuit.draw("text")
        self.assertEqual(result.__str__(), expected)
        # Extra tests that no cregbundle (or any other) warning is raised with the default settings
        # for the other drawers, if they're available to test.
        circuit.draw("latex_source")
        if optionals.HAS_MATPLOTLIB and optionals.HAS_PYLATEX:
            circuit.draw("mpl")
