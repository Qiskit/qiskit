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

from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.passmanager import PassManager
from qiskit import QuantumRegister
from qiskit.transpiler.passes import Unroller
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import CXDirection
from qiskit.transpiler.passes import SetLayout
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.transpiler.passes import FullAncillaAllocation
from qiskit.transpiler.passes import EnlargeWithAncilla
from qiskit.transpiler.passes import RemoveResetInZeroState
from qiskit.utils import optionals

from .visualization import QiskitVisualizationTestCase, path_to_diagram_reference


class TestPassManagerDrawer(QiskitVisualizationTestCase):
    """Qiskit pass manager drawer tests."""

    def setUp(self):
        super().setUp()
        coupling = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        coupling_map = CouplingMap(couplinglist=coupling)
        basis_gates = ["u1", "u3", "u2", "cx"]
        qr = QuantumRegister(7, "q")
        layout = Layout({qr[i]: i for i in range(coupling_map.size())})

        # Create a pass manager with a variety of passes and flow control structures
        self.pass_manager = PassManager()
        self.pass_manager.append(SetLayout(layout))
        self.pass_manager.append(TrivialLayout(coupling_map), condition=lambda x: True)
        self.pass_manager.append(FullAncillaAllocation(coupling_map))
        self.pass_manager.append(EnlargeWithAncilla())
        self.pass_manager.append(Unroller(basis_gates))
        self.pass_manager.append(CheckMap(coupling_map))
        self.pass_manager.append(BarrierBeforeFinalMeasurements(), do_while=lambda x: False)
        self.pass_manager.append(CXDirection(coupling_map))
        self.pass_manager.append(RemoveResetInZeroState())

    @unittest.skipIf(not optionals.HAS_GRAPHVIZ, "Graphviz not installed.")
    def test_pass_manager_drawer_basic(self):
        """Test to see if the drawer draws a normal pass manager correctly"""
        filename = "current_standard.dot"
        self.pass_manager.draw(filename=filename, raw=True)

        self.assertFilesAreEqual(filename, path_to_diagram_reference("pass_manager_standard.dot"))
        os.remove(filename)

    @unittest.skipIf(not optionals.HAS_GRAPHVIZ, "Graphviz not installed.")
    def test_pass_manager_drawer_style(self):
        """Test to see if the colours are updated when provided by the user"""
        # set colours for some passes, but leave others to take the default values
        style = {
            SetLayout: "cyan",
            CheckMap: "green",
            EnlargeWithAncilla: "pink",
            RemoveResetInZeroState: "grey",
        }

        filename = "current_style.dot"
        self.pass_manager.draw(filename=filename, style=style, raw=True)

        self.assertFilesAreEqual(filename, path_to_diagram_reference("pass_manager_style.dot"))
        os.remove(filename)


if __name__ == "__main__":
    unittest.main(verbosity=2)
