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

import unittest

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.tools.visualization import dag_drawer
from qiskit.exceptions import InvalidFileError
from qiskit.visualization.exceptions import VisualizationError
from qiskit.converters import circuit_to_dag


class TestDagDrawer(QiskitTestCase):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
