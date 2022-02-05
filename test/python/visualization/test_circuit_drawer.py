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

from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.utils import optionals
from qiskit import visualization
from qiskit.visualization import text
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
