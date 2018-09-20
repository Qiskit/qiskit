# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for qiskit/_util.py"""

import time
from qiskit.transpiler._receiver import receiver as rec
from qiskit.transpiler._parallel import parallel_map
from qiskit.transpiler._progressbar import TextProgressBar
from .common import QiskitTestCase


def _parfunc(x):
    """Function for testing parallel_map
    """
    time.sleep(1)
    return x


class TestParallel(QiskitTestCase):
    """A class for testing parallel_map functionality.
    """
    def test_parallel(self):
        """Test parallel_map """
        ans = parallel_map(_parfunc, list(range(10)))
        self.assertEqual(ans, list(range(10)))

    def test_parallel_progressbar(self):
        """Test parallel_map with progress bar"""
        TextProgressBar()
        ans = parallel_map(_parfunc, list(range(10)))
        self.assertEqual(ans, list(range(10)))

    def test_parallel_progbar_used(self):
        """Test that correct progressbar is used."""
        not_used = TextProgressBar()
        not_used.touched = True
        used = TextProgressBar()
        parallel_map(_parfunc, list(range(10)))
        self.assertTrue(used.channel_id not in rec.channels.keys())
        self.assertTrue(not_used.channel_id in rec.channels.keys())
