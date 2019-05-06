# -*- coding: utf-8 -*-

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
from qiskit.visualization.exceptions import VisualizationError
from qiskit.converters import circuit_to_dag


class TestDagDrawer(QiskitTestCase):
    """Qiskit DAG drawer tests."""

    def setUp(self):
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        self.dag = circuit_to_dag(circuit)

    def test_dag_drawer_no_graphviz(self):
        """Test dag draw with no graphviz."""
        with unittest.mock.patch('nxpd.pydot.find_graphviz', return_value=None) as _:
            self.assertRaises(VisualizationError, dag_drawer, self.dag)


if __name__ == '__main__':
    unittest.main(verbosity=2)
