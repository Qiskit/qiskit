# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Test Variational Quantum Real Time Evolution algorithm."""

import unittest

from test.python.algorithms import QiskitAlgorithmsTestCase
from numpy.testing import assert_raises
from ddt import data, ddt
import numpy as np

from qiskit.algorithms.time_evolvers.variational.var_qte import VarQTE
from qiskit.circuit import Parameter


@ddt
class TestVarQTE(QiskitAlgorithmsTestCase):
    """Test Variational Quantum Time Evolution class methods."""

    def setUp(self):
        super().setUp()
        self._parameters1 = [Parameter("a"), Parameter("b"), Parameter("c")]

    @data([1.4, 2, 3], np.asarray([1.4, 2, 3]))
    def test_create_init_state_param_dict(self, param_values):
        """Tests if a correct dictionary is created."""
        expected = dict(zip(self._parameters1, param_values))
        with self.subTest("Parameters values given as a list test."):
            result = VarQTE._create_init_state_param_dict(param_values, self._parameters1)
            np.testing.assert_equal(result, expected)
        with self.subTest("Parameters values given as a dictionary test."):
            result = VarQTE._create_init_state_param_dict(
                dict(zip(self._parameters1, param_values)), self._parameters1
            )
            np.testing.assert_equal(result, expected)
        with self.subTest("Parameters values given as a superset dictionary test."):
            expected = dict(
                zip(
                    [self._parameters1[0], self._parameters1[2]], [param_values[0], param_values[2]]
                )
            )
            result = VarQTE._create_init_state_param_dict(
                dict(zip(self._parameters1, param_values)),
                [self._parameters1[0], self._parameters1[2]],
            )
            np.testing.assert_equal(result, expected)

    @data([1.4, 2], np.asarray([1.4, 3]), {}, [])
    def test_create_init_state_param_dict_errors_list(self, param_values):
        """Tests if an error is raised."""
        with assert_raises(ValueError):
            _ = VarQTE._create_init_state_param_dict(param_values, self._parameters1)

    @data([1.4, 2], np.asarray([1.4, 3]))
    def test_create_init_state_param_dict_errors_subset(self, param_values):
        """Tests if an error is raised if subset of parameters provided."""
        param_values_dict = dict(zip([self._parameters1[0], self._parameters1[2]], param_values))
        with assert_raises(ValueError):
            _ = VarQTE._create_init_state_param_dict(param_values_dict, self._parameters1)

    @data("s")
    def test_create_init_state_param_dict_errors_value(self, param_values):
        """Tests if an error is raised if wrong input."""
        with assert_raises(ValueError):
            _ = VarQTE._create_init_state_param_dict(param_values, self._parameters1)

    @data(Parameter("x"), 5)
    def test_create_init_state_param_dict_errors_type(self, param_values):
        """Tests if an error is raised if wrong input type."""
        with assert_raises(TypeError):
            _ = VarQTE._create_init_state_param_dict(param_values, self._parameters1)


if __name__ == "__main__":
    unittest.main()
