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

""" Test NLOpt Optimizers """

import unittest
from test.aqua.common import QiskitAquaTestCase

from parameterized import parameterized
from scipy.optimize import rosen
import numpy as np

# pylint: disable=unused-import,import-outside-toplevel


class TestNLOptOptimizers(QiskitAquaTestCase):
    """ Test NLOpt Optimizers """
    def setUp(self):
        super().setUp()
        try:
            from qiskit.aqua.components.optimizers import CRS, DIRECT_L, DIRECT_L_RAND
            self.cls_dict = {CRS.__name__: CRS,
                             DIRECT_L.__name__: DIRECT_L,
                             DIRECT_L_RAND.__name__: DIRECT_L_RAND}
        except ImportError:
            self.skipTest('NLOpt dependency does not appear to be installed')
        pass

    def _optimize(self, optimizer):
        x_0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        bounds = [(-6, 6)]*len(x_0)
        res = optimizer.optimize(len(x_0), rosen, initial_point=x_0, variable_bounds=bounds)
        np.testing.assert_array_almost_equal(res[0], [1.0]*len(x_0), decimal=2)
        return res

    # ESCH and ISRES do not do well with rosen
    @parameterized.expand([
        ['CRS'],
        ['DIRECT_L'],
        ['DIRECT_L_RAND'],
    ])
    def test_nlopt(self, optimizer_name):
        """ NLopt test """
        optimizer = self.cls_dict[optimizer_name]()
        optimizer.set_options(**{'max_evals': 50000})
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 50000)


if __name__ == '__main__':
    unittest.main()
