# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test validate bounds."""

from test.python.algorithms import QiskitAlgorithmsTestCase

from unittest.mock import Mock

import numpy as np

from qiskit.algorithms.utils import validate_bounds
from qiskit.utils import algorithm_globals


class TestValidateBounds(QiskitAlgorithmsTestCase):
    """Test the ``validate_bounds`` utility function."""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 0
        self.bounds = [(-np.pi / 2, np.pi / 2)]
        self.ansatz = Mock()

    def test_with_no_ansatz_bounds(self):
        """Test with no ansatz bounds."""
        self.ansatz.num_parameters = 1
        self.ansatz.parameter_bounds = None
        bounds = validate_bounds(self.ansatz)
        self.assertEqual(bounds, [(None, None)])

    def test_with_ansatz_bounds(self):
        """Test with ansatz bounds."""
        self.ansatz.num_parameters = 1
        self.ansatz.parameter_bounds = self.bounds
        bounds = validate_bounds(self.ansatz)
        self.assertEqual(bounds, self.bounds)

    def test_with_mismatched_num_params(self):
        """Test with a mismatched number of parameters and bounds"""
        self.ansatz.num_parameters = 2
        self.ansatz.parameter_bounds = self.bounds
        with self.assertRaises(ValueError):
            _ = validate_bounds(self.ansatz)
