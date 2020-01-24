# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Tests utility functions for QuantumState classes."""

import unittest
import logging

from qiskit.test import QiskitTestCase
from qiskit.quantum_info.states import Statevector, DensityMatrix
from qiskit.quantum_info.states import partial_trace

logger = logging.getLogger(__name__)


class TestStateUtils(QiskitTestCase):
    """Test utility functions for QuantumState classes."""

    def test_statevector_partial_trace(self):
        """Test partial_trace method"""
        psi = Statevector.from_label('10+')
        self.assertEqual(
            partial_trace(psi, [0, 1]), DensityMatrix.from_label('1'))
        self.assertEqual(
            partial_trace(psi, [0, 2]), DensityMatrix.from_label('0'))
        self.assertEqual(
            partial_trace(psi, [1, 2]), DensityMatrix.from_label('+'))
        self.assertEqual(
            partial_trace(psi, [0]), DensityMatrix.from_label('10'))
        self.assertEqual(
            partial_trace(psi, [1]), DensityMatrix.from_label('1+'))
        self.assertEqual(
            partial_trace(psi, [2]), DensityMatrix.from_label('0+'))

    def test_density_matrix_partial_trace(self):
        """Test partial_trace method"""
        rho = DensityMatrix.from_label('10+')
        self.assertEqual(
            partial_trace(rho, [0, 1]), DensityMatrix.from_label('1'))
        self.assertEqual(
            partial_trace(rho, [0, 2]), DensityMatrix.from_label('0'))
        self.assertEqual(
            partial_trace(rho, [1, 2]), DensityMatrix.from_label('+'))
        self.assertEqual(
            partial_trace(rho, [0]), DensityMatrix.from_label('10'))
        self.assertEqual(
            partial_trace(rho, [1]), DensityMatrix.from_label('1+'))
        self.assertEqual(
            partial_trace(rho, [2]), DensityMatrix.from_label('0+'))


if __name__ == '__main__':
    unittest.main()
