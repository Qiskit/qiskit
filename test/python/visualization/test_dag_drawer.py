# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for DAG visualization tool."""

import os
import tempfile
import unittest

from PIL import Image

from qiskit.circuit import QuantumRegister, QuantumCircuit, Qubit, Clbit
from qiskit.tools.visualization import dag_drawer
from qiskit.exceptions import InvalidFileError
from qiskit.visualization.exceptions import VisualizationError
from qiskit.converters import circuit_to_dag
from qiskit.utils import optionals as _optionals
from .visualization import path_to_diagram_reference, QiskitVisualizationTestCase


class TestDagDrawer(QiskitVisualizationTestCase):
    """Qiskit DAG drawer tests."""

    def setUp(self):
        super().setUp()
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        self.dag = circuit_to_dag(circuit)

    def test_dag_drawer_invalid_style(self):
        """Test dag draw with invalid style."""
        self.assertRaises(VisualizationError, dag_drawer, self.dag, style="multicolor")

    def test_dag_drawer_checks_filename_correct_format(self):
        """filename must contain name and extension"""
        with self.assertRaisesRegex(
            InvalidFileError, "Parameter 'filename' must be in format 'name.extension'"
        ):
            dag_drawer(self.dag, filename="aaabc")

    def test_dag_drawer_checks_filename_extension(self):
        """filename must have a valid extension"""
        with self.assertRaisesRegex(InvalidFileError, "Filename extension must be one of: .*"):
            dag_drawer(self.dag, filename="aa.abc")

    @unittest.skipUnless(_optionals.HAS_GRAPHVIZ, "Graphviz not installed")
    def test_dag_drawer_no_register(self):
        """Test dag visualization with a circuit with no registers."""
        qubit = Qubit()
        clbit = Clbit()
        qc = QuantumCircuit([qubit, clbit])
        qc.h(0)
        qc.measure(0, 0)
        dag = circuit_to_dag(qc)
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = os.path.join(tmpdirname, "dag.png")
            dag_drawer(dag, filename=tmp_path)
            image_ref = path_to_diagram_reference("dag_no_reg.png")
            image = Image.open(tmp_path)
            self.assertImagesAreEqual(image, image_ref, 0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
