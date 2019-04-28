# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Testing for backend tools"""

from qiskit.qiskiterror import QiskitError
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeRueschlikon
from qiskit.tools.backends import reduced_coupling_map


class TestBackendTools(QiskitTestCase):
    """Backend tools tests."""

    def test_successful_reduced_map(self):
        """Generate a reduced map
        """
        fake = FakeRueschlikon()
        out = reduced_coupling_map(fake, [12, 11, 10, 9])
        ans = [[3, 2], [1, 2], [0, 1]]
        self.assertEqual(out, ans)

    def test_failed_reduced_map(self):
        """Generate a disconnected reduced map
        """
        fake = FakeRueschlikon()
        with self.assertRaises(QiskitError):
            reduced_coupling_map(fake, [1, 11, 7, 6])
