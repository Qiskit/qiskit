# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for qiskit/_util.py"""

from unittest import mock
import numpy as np

from qiskit import util
from qiskit.test import QiskitTestCase
from qiskit.utils.arithmetic import triu_to_dense


class TestUtil(QiskitTestCase):
    """Tests for qiskit/_util.py"""

    @mock.patch("platform.system", return_value="Linux")
    @mock.patch("psutil.virtual_memory")
    @mock.patch("psutil.cpu_count", return_value=None)
    def test_local_hardware_none_cpu_count(self, cpu_count_mock, vmem_mock, platform_mock):
        """Test cpu count fallback to 1 when true value can't be determined"""
        del cpu_count_mock, vmem_mock, platform_mock  # unused
        result = util.local_hardware_info()
        self.assertEqual(1, result["cpus"])

    def test_triu_to_dense(self):
        """Test conversion of upper triangular matrix to dense matrix."""
        np.random.seed(50)
        n = np.random.randint(5, 15)
        m = np.random.randint(-100, 100, size=(n, n))
        symm = (m + m.T) / 2

        triu = np.array([[symm[i, j] for i in range(j, n)] for j in range(n)])

        self.assertTrue(np.array_equal(symm, triu_to_dense(triu)))
