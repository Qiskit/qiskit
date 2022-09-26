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

"""Test validate initial point."""

from test.python.algorithms import QiskitAlgorithmsTestCase

from unittest.mock import Mock

import numpy as np

from qiskit.algorithms.utils import validate_initial_point
from qiskit.utils import algorithm_globals


class TestValidateInitialPoint(QiskitAlgorithmsTestCase):
    """Test the ``validate_initial_point`` utility function."""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 0
        self.ansatz = Mock()
        self.ansatz.num_parameters = 1

    def test_with_no_initial_point_or_bounds(self):
        """Test with no user-defined initial point and no ansatz bounds."""
        self.ansatz.parameter_bounds = None
        initial_point = validate_initial_point(None, self.ansatz)
        np.testing.assert_array_almost_equal(initial_point, [1.721111])

    def test_with_no_initial_point(self):
        """Test with no user-defined initial point with ansatz bounds."""
        self.ansatz.parameter_bounds = [(-np.pi / 2, np.pi / 2)]
        initial_point = validate_initial_point(None, self.ansatz)
        np.testing.assert_array_almost_equal(initial_point, [0.430278])

    def test_with_mismatched_params(self):
        """Test with mistmatched parameters and bounds.."""
        self.ansatz.parameter_bounds = None
        with self.assertRaises(ValueError):
            _ = validate_initial_point([1.0, 2.0], self.ansatz)
