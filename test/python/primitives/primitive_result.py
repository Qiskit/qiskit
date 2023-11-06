# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
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

from qiskit.primitives.containers import PrimitiveResult, TaskResult, make_databin
from qiskit.test import QiskitTestCase


class PrimitiveResultCase(QiskitTestCase):
    """Test the PrimitiveResult class."""

    def test_primitive_result(self):
        """Test the PrimitiveResult class."""
        data_bin_cls = make_databin(
            [("alpha", npt.NDArray[np.uint16]), ("beta", np.ndarray)], shape=(10, 20)
        )

        alpha = np.empty((10, 20), dtype=np.uint16)
        beta = np.empty((10, 20), dtype=int)

        task_results = [
            TaskResult(data_bin_cls(alpha, beta)),
            TaskResult(data_bin_cls(alpha, beta)),
        ]
        result = PrimitiveResult(task_results, {1: 2})

        self.assertTrue(result[0] is task_results[0])
        self.assertTrue(result[1] is task_results[1])
        self.assertTrue(list(result)[0] is task_results[0])
        self.assertEqual(len(result), 2)
