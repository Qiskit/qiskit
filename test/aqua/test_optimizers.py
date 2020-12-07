# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Optimizers """

import unittest
from test.aqua import QiskitAquaTestCase

from scipy.optimize import rosen
import numpy as np

from qiskit.aqua import aqua_globals
from qiskit.aqua.components.optimizers import (ADAM, CG, COBYLA, L_BFGS_B, P_BFGS, NELDER_MEAD,
                                               POWELL, SLSQP, SPSA, TNC, GSLS)


class TestOptimizers(QiskitAquaTestCase):
    """ Test Optimizers """

    def setUp(self):
        super().setUp()
        aqua_globals.random_seed = 52

    def _optimize(self, optimizer):
        x_0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        res = optimizer.optimize(len(x_0), rosen, initial_point=x_0)
        np.testing.assert_array_almost_equal(res[0], [1.0] * len(x_0), decimal=2)
        return res

    def test_adam(self):
        """ adam test """
        optimizer = ADAM(maxiter=10000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_cg(self):
        """ cg test """
        optimizer = CG(maxiter=1000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_cobyla(self):
        """ cobyla test """
        optimizer = COBYLA(maxiter=100000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 100000)

    def test_l_bfgs_b(self):
        """ l_bfgs_b test """
        optimizer = L_BFGS_B(maxfun=1000)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_p_bfgs(self):
        """ parallel l_bfgs_b test """
        optimizer = P_BFGS(maxfun=1000, max_processes=4)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_nelder_mead(self):
        """ nelder mead test """
        optimizer = NELDER_MEAD(maxfev=10000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_powell(self):
        """ powell test """
        optimizer = POWELL(maxfev=10000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_slsqp(self):
        """ slsqp test """
        optimizer = SLSQP(maxiter=1000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    @unittest.skip("Skipping SPSA as it does not do well on non-convex rozen")
    def test_spsa(self):
        """ spsa test """
        optimizer = SPSA(maxiter=10000)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 100000)

    def test_tnc(self):
        """ tnc test """
        optimizer = TNC(maxiter=1000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_gsls(self):
        """ gsls test """
        optimizer = GSLS(sample_size_factor=40, sampling_radius=1.0e-12, maxiter=10000,
                         max_eval=10000, min_step_size=1.0e-12)
        x_0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        _, x_value, n_evals = optimizer.optimize(len(x_0), rosen, initial_point=x_0)

        # Ensure value is near-optimal
        self.assertLessEqual(x_value, 0.01)
        self.assertLessEqual(n_evals, 10000)


if __name__ == '__main__':
    unittest.main()
