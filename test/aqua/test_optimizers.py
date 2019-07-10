# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import unittest

from scipy.optimize import rosen
import numpy as np

from test.common import QiskitAquaTestCase
from qiskit.aqua.components.optimizers import (ADAM, CG, COBYLA, L_BFGS_B, NELDER_MEAD,
                                               POWELL, SLSQP, SPSA, TNC)


class TestOptimizers(QiskitAquaTestCase):

    def setUp(self):
        super().setUp()
        np.random.seed(50)
        pass

    def _optimize(self, optimizer):
        x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        res = optimizer.optimize(len(x0), rosen, initial_point=x0)
        np.testing.assert_array_almost_equal(res[0], [1.0] * len(x0), decimal=2)
        return res

    def test_adam(self):
        optimizer = ADAM(maxiter=10000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_cg(self):
        optimizer = CG(maxiter=1000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_cobyla(self):
        optimizer = COBYLA(maxiter=100000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 100000)

    def test_l_bfgs_b(self):
        optimizer = L_BFGS_B(maxfun=1000)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_nelder_mead(self):
        optimizer = NELDER_MEAD(maxfev=10000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_powell(self):
        optimizer = POWELL(maxfev=10000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_slsqp(self):
        optimizer = SLSQP(maxiter=1000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    @unittest.skip("Skipping SPSA as it does not do well on non-convex rozen")
    def test_spsa(self):
        optimizer = SPSA(max_trials=10000)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 100000)

    def test_tnc(self):
        optimizer = TNC(maxiter=1000, tol=1e-06)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)


if __name__ == '__main__':
    unittest.main()
