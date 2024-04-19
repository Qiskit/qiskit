# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Unit tests for PrimitiveResult."""

import numpy as np
import numpy.typing as npt

from qiskit.primitives.containers import PrimitiveResult, PubResult, make_data_bin
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class PrimitiveResultCase(QiskitTestCase):
    """Test the PrimitiveResult class."""

    def test_primitive_result(self):
        """Test the PrimitiveResult class."""
        data_bin_cls = make_data_bin(
            [("alpha", npt.NDArray[np.uint16]), ("beta", np.ndarray)], shape=(10, 20)
        )

        alpha = np.empty((10, 20), dtype=np.uint16)
        beta = np.empty((10, 20), dtype=int)

        pub_results = [
            PubResult(data_bin_cls(alpha=alpha, beta=beta)),
            PubResult(data_bin_cls(alpha=alpha, beta=beta)),
        ]
        result = PrimitiveResult(pub_results, {"x": 2})

        self.assertTrue(result[0] is pub_results[0])
        self.assertTrue(result[1] is pub_results[1])
        self.assertTrue(list(result)[0] is pub_results[0])
        self.assertEqual(len(result), 2)
        self.assertEqual(result.metadata, {"x": 2})
