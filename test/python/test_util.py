# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for qiskit/utils"""

from unittest import mock

from qiskit.utils import multiprocessing
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestUtil(QiskitTestCase):
    """Tests for qiskit/_util.py"""

    def test_local_hardware_five_cpu_count(self):
        """Test cpu count is half when sched affinity is 5"""
        with mock.patch.object(multiprocessing, "os"):
            multiprocessing.os.sched_getaffinity = mock.MagicMock(return_value=set(range(5)))
            result = multiprocessing.local_hardware_info()
        self.assertEqual(2, result["cpus"])

    def test_local_hardware_sixty_four_cpu_count(self):
        """Test cpu count is 32 when sched affinity is 64"""
        with mock.patch.object(multiprocessing, "os"):
            multiprocessing.os.sched_getaffinity = mock.MagicMock(return_value=set(range(64)))
            result = multiprocessing.local_hardware_info()
        self.assertEqual(32, result["cpus"])

    def test_local_hardware_no_cpu_count(self):
        """Test cpu count fallback to 1 when true value can't be determined"""
        with mock.patch.object(multiprocessing, "os"):
            multiprocessing.os.sched_getaffinity = mock.MagicMock(return_value=set())
            result = multiprocessing.local_hardware_info()
        self.assertEqual(1, result["cpus"])

    def test_local_hardware_no_sched_five_count(self):
        """Test cpu could if sched affinity method is missing and cpu count is 5."""
        with mock.patch.object(multiprocessing, "os", spec=[]):
            multiprocessing.os.cpu_count = mock.MagicMock(return_value=5)
            del multiprocessing.os.sched_getaffinity
            result = multiprocessing.local_hardware_info()
        self.assertEqual(2, result["cpus"])

    def test_local_hardware_no_sched_sixty_four_count(self):
        """Test cpu could if sched affinity method is missing and cpu count is 64."""
        with mock.patch.object(multiprocessing, "os", spec=[]):
            multiprocessing.os.cpu_count = mock.MagicMock(return_value=64)
            del multiprocessing.os.sched_getaffinity
            result = multiprocessing.local_hardware_info()
        self.assertEqual(32, result["cpus"])

    def test_local_hardware_no_sched_no_count(self):
        """Test cpu count fallback to 1 when no sched getaffinity available."""
        with mock.patch.object(multiprocessing, "os", spec=[]):
            multiprocessing.os.cpu_count = mock.MagicMock(return_value=None)
            del multiprocessing.os.sched_getaffinity
            result = multiprocessing.local_hardware_info()
        self.assertEqual(1, result["cpus"])
