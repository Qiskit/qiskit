# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for qiskit/util .py"""

from qiskit.util import _check_ibmqx_version
from .common import QiskitTestCase


class TestUtil(QiskitTestCase):
    """Tests for qiskit/util .py"""

    def test_check_ibmqx_version(self):
        """Required IBMQE version."""
        with self.assertNoLogs('qiskit.util ', level='WARNING'):
            _check_ibmqx_version()
