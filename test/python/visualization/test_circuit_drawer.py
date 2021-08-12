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
from PIL import Image

from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit import visualization
from qiskit.visualization import text
from qiskit.visualization.exceptions import VisualizationError

if visualization.HAS_MATPLOTLIB:
    from matplotlib import figure


class TestCircuitDrawer(QiskitTestCase):
    def test_default_output(self):
        with patch("qiskit.user_config.get_config", return_value={}):
            circuit = QuantumCircuit()
            out = visualization.circuit_drawer(circuit)
            self.assertIsInstance(out, text.TextDrawing)

    @unittest.skipUnless(
        visualization.HAS_MATPLOTLIB, "Skipped because matplotlib is not available"
    )
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

    @unittest.skipUnless(
        visualization.HAS_MATPLOTLIB, "Skipped because matplotlib is not available"
    )
    def test_kwarg_priority_over_user_config_default_output(self):
        with patch("qiskit.user_config.get_config", return_value={"circuit_drawer": "latex"}):
            circuit = QuantumCircuit()
            out = visualization.circuit_drawer(circuit, output="mpl")
            self.assertIsInstance(out, figure.Figure)

    @unittest.skipUnless(
        visualization.HAS_MATPLOTLIB, "Skipped because matplotlib is not available"
    )
    def test_default_backend_auto_output_with_mpl(self):
        with patch("qiskit.user_config.get_config", return_value={"circuit_drawer": "auto"}):
            circuit = QuantumCircuit()
            out = visualization.circuit_drawer(circuit)
            self.assertIsInstance(out, figure.Figure)

    def test_default_backend_auto_output_without_mpl(self):
        with patch("qiskit.user_config.get_config", return_value={"circuit_drawer": "auto"}):
            with patch.object(
                visualization.circuit_visualization, "_matplotlib", autospec=True
            ) as mpl_mock:
                mpl_mock.HAS_MATPLOTLIB = False
                circuit = QuantumCircuit()
                out = visualization.circuit_drawer(circuit)
                self.assertIsInstance(out, text.TextDrawing)

    @unittest.skipUnless(
        visualization.HAS_PYLATEX and visualization.HAS_PIL and visualization.HAS_PDFLATEX,
        "Skipped because PIL and pylatex is not available",
    )
    def test_unsupported_image_format_error_message(self):
        with patch("qiskit.user_config.get_config", return_value={"circuit_drawer": "latex"}):
            circuit = QuantumCircuit()
            with self.assertRaises(VisualizationError) as ve:
                visualization.circuit_drawer(circuit, filename="file.spooky")
                self.assertEqual(
                    str(ve.exception),
                    "ERROR: filename parameter does not use a supported extension.",
                )

    @unittest.skipUnless(
        visualization.HAS_PYLATEX and visualization.HAS_PIL and visualization.HAS_PDFLATEX,
        "Skipped because PIL and pylatex is not available",
    )
    def test_output_file_correct_format(self):
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
