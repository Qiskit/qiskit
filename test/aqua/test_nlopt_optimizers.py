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

from parameterized import parameterized
from scipy.optimize import rosen
import numpy as np

from test.common import QiskitAquaTestCase
from qiskit.aqua import PluggableType, get_pluggable_class


class TestNLOptOptimizers(QiskitAquaTestCase):

    def setUp(self):
        super().setUp()
        np.random.seed(50)
        try:
            import nlopt
        except ImportError:
            self.skipTest('NLOpt dependency does not appear to be installed')
        pass

    def _optimize(self, optimizer):
        x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        bounds = [(-6, 6)]*len(x0)
        res = optimizer.optimize(len(x0), rosen, initial_point=x0, variable_bounds=bounds)
        np.testing.assert_array_almost_equal(res[0], [1.0]*len(x0), decimal=2)
        return res

    # ESCH and ISRES do not do well with rosen
    @parameterized.expand([
        ['CRS'],
        ['DIRECT_L'],
        ['DIRECT_L_RAND'],
        # ['ESCH'],
        # ['ISRES']
    ])
    def test_nlopt(self, name):
        optimizer = get_pluggable_class(PluggableType.OPTIMIZER, name)()
        optimizer.set_options(**{'max_evals': 50000})
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 50000)


if __name__ == '__main__':
    unittest.main()
