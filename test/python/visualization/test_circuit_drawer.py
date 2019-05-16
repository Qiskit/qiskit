# -*- coding: utf-8 -*-

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

from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit import visualization
from qiskit.visualization import text

if visualization.HAS_MATPLOTLIB:
    from matplotlib import figure


class TestCircuitDrawer(QiskitTestCase):

    def test_default_output(self):
        with unittest.mock.patch('qiskit.user_config.get_config',
                                 return_value={}):
            circuit = QuantumCircuit()
            out = visualization.circuit_drawer(circuit)
            self.assertIsInstance(out, text.TextDrawing)

    @unittest.skipUnless(visualization.HAS_MATPLOTLIB,
                         'Skipped because matplotib is not available')
    def test_user_config_default_output(self):
        with unittest.mock.patch('qiskit.user_config.get_config',
                                 return_value={'circuit_drawer': 'mpl'}):
            circuit = QuantumCircuit()
            out = visualization.circuit_drawer(circuit)
            self.assertIsInstance(out, figure.Figure)

    def test_default_output_with_user_config_not_set(self):
        with unittest.mock.patch('qiskit.user_config.get_config',
                                 return_value={'other_option': True}):
            circuit = QuantumCircuit()
            out = visualization.circuit_drawer(circuit)
            self.assertIsInstance(out, text.TextDrawing)

    @unittest.skipUnless(visualization.HAS_MATPLOTLIB,
                         'Skipped because matplotib is not available')
    def test_kwarg_priority_over_user_config_default_output(self):
        with unittest.mock.patch('qiskit.user_config.get_config',
                                 return_value={'circuit_drawer': 'latex'}):
            circuit = QuantumCircuit()
            out = visualization.circuit_drawer(circuit, output='mpl')
            self.assertIsInstance(out, figure.Figure)

    @unittest.skipUnless(visualization.HAS_MATPLOTLIB,
                         'Skipped because matplotib is not available')
    def test_default_backend_auto_output_with_mpl(self):
        with unittest.mock.patch('qiskit.user_config.get_config',
                                 return_value={'circuit_drawer': 'auto'}):
            circuit = QuantumCircuit()
            out = visualization.circuit_drawer(circuit)
            self.assertIsInstance(out, figure.Figure)

    def test_default_backend_auto_output_without_mpl(self):
        with unittest.mock.patch('qiskit.user_config.get_config',
                                 return_value={'circuit_drawer': 'auto'}):
            with unittest.mock.patch.object(
                    visualization.circuit_visualization, '_matplotlib',
                    autospec=True) as mpl_mock:
                mpl_mock.HAS_MATPLOTLIB = False
                circuit = QuantumCircuit()
                out = visualization.circuit_drawer(circuit)
                self.assertIsInstance(out, text.TextDrawing)
