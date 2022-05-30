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

"""Tests for the UMDA optimizer."""

from test.python.algorithms import QiskitAlgorithmsTestCase

import numpy as np
from qiskit.algorithms.optimizers.umda import UMDA


class TestUMDA(QiskitAlgorithmsTestCase):
    """Tests for the UMDA optimizer."""

    def test_truncation(self):
        """Test if truncation is performed as expected"""
        umda = UMDA(maxiter=1, size_gen=20, n_variables=10)

        def objective_function(array):
            return sum(array)

        umda.minimize(objective_function, np.array([np.pi] * 10))

        siz = int(umda.ELITE_FACTOR * int(umda.size_gen * umda.alpha))
        assert umda.generation.shape == (siz + umda.size_gen, umda.n_variables)

    def test_get_set(self):
        """Test if getters and setters work as expected"""
        umda = UMDA(maxiter=1, size_gen=20, n_variables=10)
        umda.disp = True
        umda.size_gen = 30
        umda.alpha = 0.6
        umda.dead_iter = 10
        umda.max_iter = 100

        assert umda.disp is True
        assert umda.size_gen == 30
        assert umda.alpha == 0.6
        assert umda.dead_iter == 10
        assert umda.max_iter == 100

    def test_settings(self):
        """Test if the settings display works well"""
        umda = UMDA(maxiter=1, size_gen=20, n_variables=10)
        umda.disp = True
        umda.size_gen = 30
        umda.alpha = 0.6
        umda.dead_iter = 10
        umda.max_iter = 100

        set_ = {
            "max_iter": 100,
            "alpha": 0.6,
            "dead_iter": 10,
            "size_gen": 30,
            "n_variables": 10,
            "best_cost_global": umda.best_cost_global,
            "best_ind_global": umda.best_ind_global,
        }

        assert umda.settings == set_
