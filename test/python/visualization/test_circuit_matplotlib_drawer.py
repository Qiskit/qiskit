# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring

import os
import tempfile
import unittest

from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit import visualization

if visualization.HAS_MATPLOTLIB:
    from matplotlib import pyplot as plt
    from matplotlib.testing import compare


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

    @unittest.skipIf(not visualization.HAS_MATPLOTLIB,
                     'matplotlib not available.')
    @unittest.skipIf(os.name == 'nt', 'tempfile fails on appveyor')
    def test_empty_circuit(self):
        qc = QuantumCircuit()
        res = visualization.circuit_drawer(qc, output='mpl')
        res_out_file = tempfile.NamedTemporaryFile(suffix='.png')
        self.addCleanup(res_out_file.close)
        res.savefig(res_out_file.name)
        expected = self._expected_empty()
        expected_image_file = tempfile.NamedTemporaryFile(suffix='.png')
        self.addCleanup(expected_image_file.close)
        expected.savefig(expected_image_file.name)
        self.assertIsNone(compare.compare_images(expected_image_file.name,
                                                 res_out_file.name, 0.0001))
