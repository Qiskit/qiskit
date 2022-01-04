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
"""Tests GibbsState class."""
import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np

from qiskit.algorithms.gibbs_state_preparation.gibbs_state import GibbsState
from qiskit.opflow import Zero, X


class TestGibbsState(QiskitAlgorithmsTestCase):
    """Tests GibbsState class."""

    def test_gibbs_state_init(self):
        """Initialization test."""
        gibbs_state_function = Zero
        hamiltonian = X
        temperature = 42

        gibbs_state = GibbsState(gibbs_state_function, hamiltonian, temperature)

        np.testing.assert_equal(gibbs_state.gibbs_state_function, Zero)
        np.testing.assert_equal(gibbs_state.hamiltonian, X)
        np.testing.assert_equal(gibbs_state.temperature, 42)
        np.testing.assert_equal(gibbs_state.gradients, None)
        np.testing.assert_equal(gibbs_state.gradient_params, None)


if __name__ == "__main__":
    unittest.main()
