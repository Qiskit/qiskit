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

""" Test NLOpt Optimizers """

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
from ddt import ddt, idata, unpack
from scipy.optimize import rosen
import numpy as np

from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.algorithms.optimizers import CRS, DIRECT_L, DIRECT_L_RAND


@ddt
class TestNLOptOptimizers(QiskitAlgorithmsTestCase):
    """Test NLOpt Optimizers"""

    def _optimize(self, optimizer, use_bound):
        x_0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        bounds = [(-6, 6)] * len(x_0) if use_bound else None
        res = optimizer.optimize(len(x_0), rosen, initial_point=x_0, variable_bounds=bounds)
        np.testing.assert_array_almost_equal(res[0], [1.0] * len(x_0), decimal=2)
        return res

    # ESCH and ISRES do not do well with rosen
    @idata(
        [
            [CRS, True],
            [DIRECT_L, True],
            [DIRECT_L_RAND, True],
            [CRS, False],
            [DIRECT_L, False],
            [DIRECT_L_RAND, False],
        ]
    )
    @unpack
    def test_nlopt(self, optimizer_cls, use_bound):
        """NLopt test"""
        try:
            optimizer = optimizer_cls()
            optimizer.set_options(**{"max_evals": 50000})
            res = self._optimize(optimizer, use_bound)
            self.assertLessEqual(res[2], 50000)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))


if __name__ == "__main__":
    unittest.main()
