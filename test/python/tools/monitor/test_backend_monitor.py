# -*- coding: utf-8 -*-

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

"""Tests for the wrapper functionality."""

import unittest
from unittest.mock import patch
from io import StringIO

from qiskit.tools.monitor import backend_overview, backend_monitor
from qiskit.test import QiskitTestCase, online_test


class TestBackendOverview(QiskitTestCase):
    """Tools test case."""

    @online_test
    def test_backend_overview(self, qe_token, qe_url):
        """Test backend_overview"""
        from qiskit import IBMQ  # pylint: disable: import-error
        IBMQ.enable_account(qe_token, qe_url)
        self.addCleanup(IBMQ.disable_account)

        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            backend_overview()
        stdout = fake_stdout.getvalue()
        self.assertIn('Operational:', stdout)
        self.assertIn('Avg. T1:', stdout)
        self.assertIn('Num. Qubits:', stdout)

    @online_test
    def test_backend_monitor(self, qe_token, qe_url):
        """Test backend_monitor"""
        from qiskit import IBMQ  # pylint: disable: import-error
        IBMQ.enable_account(qe_token, qe_url)
        self.addCleanup(IBMQ.disable_account)

        for back in IBMQ.backends():
            if not back.configuration().simulator:
                backend = back
                break
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            backend_monitor(backend)

        stdout = fake_stdout.getvalue()
        self.assertIn('Configuration', stdout)
        self.assertIn('Qubits [Name / Freq / T1 / T2 / U1 err / U2 err / U3 err / Readout err]',
                      stdout)
        self.assertIn('Multi-Qubit Gates [Name / Type / Gate Error]', stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
