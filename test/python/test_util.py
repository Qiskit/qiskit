# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for qiskit/_util.py"""

from unittest import mock

from qiskit import util
from qiskit.test import QiskitTestCase


class TestUtil(QiskitTestCase):
    """Tests for qiskit/_util.py"""

    @mock.patch('platform.system', return_value='Linux')
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.cpu_count', return_value=None)
    def test_local_hardware_none_cpu_count(self, cpu_count_mock, vmem_mock,
                                           platform_mock):
        """Test cpu count fallback to 1 when true value can't be determined"""
        # pylint: disable=unused-argument
        result = util.local_hardware_info()
        self.assertEqual(1, result['cpus'])
