# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring

import tempfile
import unittest
import os

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.test import QiskitTestCase
from qiskit import visualization


if visualization.HAS_MATPLOTLIB:
    from matplotlib import pyplot as plt
    from matplotlib.testing import compare
    import matplotlib
    import sys


def _path_to_reference(filename):
    return os.path.join(_this_directory(), 'references', filename)


def _this_directory():
    return os.path.dirname(os.path.abspath(__file__))


class TestMatplotlibDrawer(QiskitTestCase):

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

    def _make_temp_file(self, plot):
        tmp = tempfile.NamedTemporaryFile(suffix='.png')
        self.addCleanup(tmp.close)
        plot.savefig(tmp.name)
        return tmp

    @unittest.skipIf(not visualization.HAS_MATPLOTLIB,
                     'matplotlib not available.')
    def test_empty_circuit(self):
        qc = QuantumCircuit()
        res = visualization.circuit_drawer(qc, output='mpl')
        res_out_file = self._make_temp_file(res)
        expected = self._expected_empty()
        expected_image_file = self._make_temp_file(expected)
        self.assertIsNone(compare.compare_images(expected_image_file.name,
                                                 res_out_file.name, 0.0001))

    @unittest.skipIf(not visualization.HAS_MATPLOTLIB,
                     'matplotlib not available.')
    def test_plot_barriers(self):
        """Test to see that plotting barriers works - if it is set to False, no
        blank columns are introduced"""

        # Use a different backend as the default backend causes the test to fail.
        # This is because it adds less whitespace around the image than is present
        # in the reference image, but only on MacOS
        if sys.platform == 'darwin':
            matplotlib.use('agg')

        # generate a circuit with barriers and other barrier like instructions in
        q = QuantumRegister(2, 'q')
        c = ClassicalRegister(2, 'c')
        qc = QuantumCircuit(q, c)

        # check for barriers
        qc.h(q[0])
        qc.barrier()

        # check for other barrier like commands
        qc.h(q[1])

        # this import appears to be unused, but is actually needed to get snapshot instruction
        import qiskit.extensions.simulator  # pylint: disable=unused-import
        qc.snapshot('1')

        # check the barriers plot properly when plot_barriers= True
        barriers_plot = visualization.circuit_drawer(qc, output='mpl', plot_barriers=True)
        barriers_plot_file = self._make_temp_file(barriers_plot)
        self.assertIsNone(compare.compare_images(barriers_plot_file.name,
                                                 _path_to_reference('matplotlib_barriers_ref.png'),
                                                 0.0001))

        # check that the barrier aren't plotted when plot_barriers = False
        barriers_no_plot = visualization.circuit_drawer(qc, output='mpl', plot_barriers=False)
        barriers_no_plot_file = self._make_temp_file(barriers_no_plot)

        # generate the same circuit but without the barrier commands as this is what the
        # circuit should look like when displayed with plot barriers false
        q1 = QuantumRegister(2, 'q')
        c1 = ClassicalRegister(2, 'c')
        qc1 = QuantumCircuit(q1, c1)
        qc1.h(q1[0])
        qc1.h(q1[1])

        no_barriers = visualization.circuit_drawer(qc1, output='mpl', justify='None',)
        no_barriers_file = self._make_temp_file(no_barriers)

        self.assertIsNone(compare.compare_images(barriers_no_plot_file.name,
                                                 no_barriers_file.name, 0.0001))
