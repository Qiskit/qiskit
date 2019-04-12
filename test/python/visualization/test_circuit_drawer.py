# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=no-member,invalid-name,missing-docstring

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
