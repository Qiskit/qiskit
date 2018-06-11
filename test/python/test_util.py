# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for qiskit/_util.py"""

import unittest
from qiskit._util import _check_ibmqx_version
from .common import QiskitTestCase


class TestUtil(QiskitTestCase):
    """Tests for qiskit/_util.py"""

    @unittest.skip("Temporary skipping")
    def test_check_ibmqx_version(self):
        """Requiered IBMQE version."""
        with self.assertNoLogs('qiskit._util', level='WARNING'):
            _check_ibmqx_version()
