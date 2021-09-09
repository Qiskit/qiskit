# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
import unittest

import numpy as np

from qiskit.algorithms.quantum_time_evolution.variational.error_calculators.time_step_errors.time_step_error_calculator import (
    _calculate_error_term,
    _calculate_max_bures,
    _calculate_energy_factor,
)
from test.python.algorithms import QiskitAlgorithmsTestCase


class TestTimeStepErrorCalculator(QiskitAlgorithmsTestCase):
    def test_calculate_error_term(self):
        # data below is dummy
        d_t, eps_t, grad_err, energy, h_squared, h_norm, stddev = 1.2, 4.3, 3.3, 1.3, 5.34, 2, 5
        error_term = _calculate_error_term(d_t, eps_t, grad_err, energy, h_squared, h_norm, stddev)

        expected_error_term = 68.07010673872762

        np.testing.assert_equal(error_term, expected_error_term)

    def test_calculate_max_bures(self):
        # data below is dummy
        eps, e, e_factor, h_squared, delta_t = 1.2, 4.3, 3.3, 1.3, 5.34
        max_bures = _calculate_max_bures(eps, e, e_factor, h_squared, delta_t)

        expected_max_bures = 6.102757696751
        np.testing.assert_equal(max_bures, expected_max_bures)

    def test_calculate_energy_factor(self):
        # data below is dummy
        eps_t, energy, stddev, h_norm = 1.2, 4.3, 3.3, 1.3
        energy_factor = _calculate_energy_factor(eps_t, energy, stddev, h_norm)

        expected_energy_factor = 4.112664186294103
        np.testing.assert_equal(energy_factor, expected_energy_factor)


if __name__ == "__main__":
    unittest.main()
