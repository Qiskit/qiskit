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

"""Tests for pass manager visualization tool."""

import unittest
import os

from qiskit.transpiler.transpile_config import TranspileConfig
from qiskit.transpiler import CouplingMap, Layout
from qiskit import QuantumRegister
from qiskit.transpiler.preset_passmanagers import level_0_pass_manager, level_1_pass_manager
from qiskit.transpiler.passes import SetLayout, CheckMap, EnlargeWithAncilla, RemoveResetInZeroState

from qiskit.visualization.pass_manager_visualization import HAS_GRAPHVIZ
from .visualization import QiskitVisualizationTestCase, path_to_diagram_reference


class TestPassManagerDrawer(QiskitVisualizationTestCase):
    """Qiskit pass manager drawer tests."""

    def setUp(self):
        coupling = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        coupling_map = CouplingMap(couplinglist=coupling)
        qr = QuantumRegister(7, 'q')
        layout = Layout({qr[i]: i for i in range(coupling_map.size())})
        self.config = TranspileConfig(optimization_level=1,
                                      basis_gates=['u1', 'u3', 'u2', 'cx'],
                                      initial_layout=layout,
                                      coupling_map=coupling_map,
                                      seed_transpiler=987,
                                      backend_properties=None)

    @unittest.skipIf(not HAS_GRAPHVIZ,
                     'Graphviz not installed.')
    def test_pass_manager_drawer_basic(self):
        """Test to see if the drawer draws a normal pass manager correctly"""
        filename = self._get_resource_path('current_l0.dot')
        level_0_pass_manager(self.config).draw(filename=filename, raw=True)

        self.assertDotFilesAreEqual(filename, path_to_diagram_reference('pass_manager_l0.dot'))
        os.remove(filename)

    @unittest.skipIf(not HAS_GRAPHVIZ,
                     'Graphviz not installed.')
    def test_pass_manager_drawer_style(self):
        """Test to see if the colours are updated when provided by the user"""
        # set colours for some passes, but leave others to take the default values
        style = {SetLayout: 'cyan',
                 CheckMap: 'green',
                 EnlargeWithAncilla: 'pink',
                 RemoveResetInZeroState: 'grey'}

        filename = self._get_resource_path('current_l1.dot')
        level_1_pass_manager(self.config).draw(filename=filename, style=style, raw=True)

        self.assertDotFilesAreEqual(filename,
                                    path_to_diagram_reference('pass_manager_style_l1.dot'))
        os.remove(filename)

    def assertDotFilesAreEqual(self, current, expected):
        """
        Asserts that 2 dot files are the same - can't use a straightforward
        line by line comparison as clusters are named randomly by pydot and
        so the files will always differ as the cluster names will be different
        """

        self.assertTrue(os.path.exists(current))
        self.assertTrue(os.path.exists(expected))

        with open(current) as curr_file, open(expected) as exp_file:
            curr_lines = curr_file.readlines()
            exp_lines = exp_file.readlines()

            self.assertTrue(len(curr_lines) == len(exp_lines))

            for index, curr_line in enumerate(curr_lines):
                exp_line = exp_lines[index]

                # if the lines are the same continue checking
                if curr_line == exp_line:
                    continue

                # the only way lines can be different is if they are declaring the subgraph
                # as these include a randomly generated id
                self.assertTrue(curr_line.startswith('subgraph')
                                and exp_line.startswith('subgraph'))

        # all checks have passed so dot files must be equal


if __name__ == '__main__':
    unittest.main(verbosity=2)
