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

# pylint: disable=invalid-name,missing-docstring

import unittest
import os

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import visualization

from .visualization import QiskitVisualizationTestCase

if visualization.HAS_MATPLOTLIB:
    from matplotlib import pyplot as plt


def _path_to_reference(filename):
    return os.path.join(_this_directory(), 'references', filename)


def _this_directory():
    return os.path.dirname(os.path.abspath(__file__))


class TestMatplotlibDrawer(QiskitVisualizationTestCase):

    def _expected_empty(self):
        # Generate blank
        expected = plt.figure()
        expected.patch.set_facecolor(color='#ffffff')
        ax = expected.add_subplot(111)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.tick_params(labelbottom=False, labeltop=False,
                       labelleft=False, labelright=False)
        expected.set_size_inches(2.508333333333333, 0.2508333333333333)
        return expected

    @unittest.skipIf(not visualization.HAS_MATPLOTLIB, 'matplotlib not available.')
    def test_empty_circuit(self):
        qc = QuantumCircuit()
        filename = self._get_resource_path('current_pulse_matplotlib_ref.png')
        visualization.circuit_drawer(qc, output='mpl', filename=filename)
        self.addCleanup(os.remove, filename)

        expected_filename = self._get_resource_path('expected_current_pulse_matplotlib_ref.png')
        expected = self._expected_empty()
        expected.savefig(expected_filename)
        self.addCleanup(os.remove, expected_filename)

        self.assertImagesAreEqual(filename, expected_filename)

    @unittest.skipIf(not visualization.HAS_MATPLOTLIB,
                     'matplotlib not available.')
    @unittest.skip('Unreliable across python version')
    def test_long_name(self):
        """Test to see that long register names can be seen completely
        As reported in #2605
        """

        # add a register with a very long name
        qr = QuantumRegister(4, 'veryLongQuantumRegisterName')
        # add another to make sure adjustments are made based on longest
        qrr = QuantumRegister(1, 'q0')
        circuit = QuantumCircuit(qr, qrr)

        # check gates are shifted over accordingly
        circuit.h(qr)
        circuit.h(qr)
        circuit.h(qr)

        filename = self._get_resource_path('current_%s_long_name_matplotlib.png' % os.name)
        visualization.circuit_drawer(circuit, output='mpl', filename=filename)
        self.addCleanup(os.remove, filename)

        ref_filename = self._get_resource_path(
            'visualization/references/%s_long_name_matplotlib.png' % os.name)

        self.assertImagesAreEqual(ref_filename, filename)
