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

    @unittest.skipIf(not visualization.HAS_MATPLOTLIB, 'matplotlib not available.')
    def test_plot_barriers(self):
        """Test to see that plotting barriers works.
        If it is set to False, no blank columns are introduced"""

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
        filename = self._get_resource_path('visualization/references/current_matplotlib_ref.png')
        visualization.circuit_drawer(qc, output='mpl', plot_barriers=True, filename=filename)
        self.addCleanup(os.remove, filename)

        ref_filename = self._get_resource_path(
            'visualization/references/matplotlib_barriers_ref.png')
        self.assertImagesAreEqual(filename, ref_filename)

        # check that the barrier aren't plotted when plot_barriers = False
        filename = self._get_resource_path('current_matplotlib_ref.png')
        visualization.circuit_drawer(qc, output='mpl', plot_barriers=False, filename=filename)
        self.addCleanup(os.remove, filename)

        # generate the same circuit but without the barrier commands as this is what the
        # circuit should look like when displayed with plot barriers false
        q1 = QuantumRegister(2, 'q')
        c1 = ClassicalRegister(2, 'c')
        qc1 = QuantumCircuit(q1, c1)
        qc1.h(q1[0])
        qc1.h(q1[1])

        no_barriers_filename = self._get_resource_path('current_no_barriers_matplotlib_ref.png')
        visualization.circuit_drawer(qc1, output='mpl', justify='None',
                                     filename=no_barriers_filename)
        self.addCleanup(os.remove, no_barriers_filename)

        self.assertImagesAreEqual(filename, no_barriers_filename)

    @unittest.skipIf(not visualization.HAS_MATPLOTLIB,
                     'matplotlib not available.')
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
        # self.addCleanup(os.remove, filename)

        ref_filename = self._get_resource_path(
            'visualization/references/%s_long_name_matplotlib.png' % os.name)

        self.assertImagesAreEqual(ref_filename, filename)

    @unittest.skipIf(not visualization.HAS_MATPLOTLIB,
                     'matplotlib not available.')
    def test_conditional(self):
        """Test that circuits with conditionals draw correctly
        """
        qr = QuantumRegister(2, 'q')
        cr = ClassicalRegister(2, 'c')
        circuit = QuantumCircuit(qr, cr)

        # check gates are shifted over accordingly
        circuit.h(qr)
        circuit.measure(qr, cr)
        circuit.h(qr[0]).c_if(cr, 2)

        conditional_filename = self._get_resource_path('current_conditional_matplotlib_ref.png')
        visualization.circuit_drawer(circuit, output='mpl',
                                     filename=conditional_filename)
        self.addCleanup(os.remove, conditional_filename)

        ref_filename = self._get_resource_path(
            'visualization/references/matplotlib_conditional_ref.png')

        self.assertImagesAreEqual(ref_filename, conditional_filename)
