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

from qiskit.algorithms.optimizers.umda import UMDA


class TestUMDA(QiskitAlgorithmsTestCase):
    """Tests for the UMDA optimizer."""

    def test_get_set(self):
        """Test if getters and setters work as expected"""
        umda = UMDA(maxiter=1, size_gen=20)
        umda.disp = True
        umda.size_gen = 30
        umda.alpha = 0.6
        umda.maxiter = 100

        assert umda.disp is True
        assert umda.size_gen == 30
        assert umda.alpha == 0.6
        assert umda.maxiter == 100

    def test_settings(self):
        """Test if the settings display works well"""
        umda = UMDA(maxiter=1, size_gen=20)
        umda.disp = True
        umda.size_gen = 30
        umda.alpha = 0.6
        umda.maxiter = 100

        set_ = {
            "maxiter": 100,
            "alpha": 0.6,
            "size_gen": 30,
        }

        assert umda.settings == set_

    def test_minimize(self):
        """optimize function test"""
        from scipy.optimize import rosen
        import numpy as np
        from qiskit.utils import algorithm_globals

        # UMDA is volatile so we need to set the seeds for the execution
        algorithm_globals.random_seed = 52

        optimizer = UMDA(maxiter=1000, size_gen=100)
        x_0 = [1.3, 0.7, 1.5]
        res = optimizer.minimize(rosen, x_0)

        assert res.fun is not None
        assert len(res.x) == len(x_0)

        np.testing.assert_array_almost_equal(res.x, [1.0] * len(x_0), decimal=2)
