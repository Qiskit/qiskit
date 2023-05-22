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
from scipy.optimize import rosen

from qiskit.algorithms.optimizers.umda import UMDA
from qiskit.utils import algorithm_globals


class TestUMDA(QiskitAlgorithmsTestCase):
    """Tests for the UMDA optimizer."""

    def test_get_set(self):
        """Test if getters and setters work as expected"""
        umda = UMDA(maxiter=1, size_gen=20)
        umda.disp = True
        umda.size_gen = 30
        umda.alpha = 0.6
        umda.maxiter = 100

        self.assertTrue(umda.disp)
        self.assertEqual(umda.size_gen, 30)
        self.assertEqual(umda.alpha, 0.6)
        self.assertEqual(umda.maxiter, 100)

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
            "callback": None,
        }

        self.assertEqual(umda.settings, set_)

    def test_minimize(self):
        """optimize function test"""
        # UMDA is volatile so we need to set the seeds for the execution
        algorithm_globals.random_seed = 52

        optimizer = UMDA(maxiter=1000, size_gen=100)
        x_0 = [1.3, 0.7, 1.5]
        res = optimizer.minimize(rosen, x_0)

        self.assertIsNotNone(res.fun)
        self.assertEqual(len(res.x), len(x_0))
        np.testing.assert_array_almost_equal(res.x, [1.0] * len(x_0), decimal=2)

    def test_callback(self):
        """Test the callback."""

        def objective(x):
            return np.linalg.norm(x) - 1

        nfevs, parameters, fvals = [], [], []

        def store_history(*args):
            nfevs.append(args[0])
            parameters.append(args[1])
            fvals.append(args[2])

        optimizer = UMDA(maxiter=1, callback=store_history)
        _ = optimizer.minimize(objective, x0=np.arange(5))

        self.assertEqual(len(nfevs), 1)
        self.assertIsInstance(nfevs[0], int)

        self.assertEqual(len(parameters), 1)
        self.assertIsInstance(parameters[0], np.ndarray)
        self.assertEqual(parameters[0].size, 5)

        self.assertEqual(len(fvals), 1)
        self.assertIsInstance(fvals[0], float)
