# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for the wrapper functionality."""

import unittest
from unittest.mock import patch
from io import StringIO

from qiskit.tools.monitor import backend_overview, backend_monitor
from qiskit.test import QiskitTestCase, requires_qe_access


class TestBackendOverview(QiskitTestCase):
    """Tools test case."""

    @requires_qe_access
    def test_backend_overview(self, qe_token, qe_url):
        """Test backend_overview"""
        from qiskit import IBMQ  # pylint: disable: import-error
        IBMQ.enable_account(qe_token, qe_url)

        with patch('sys.stdout', new=StringIO()) as fake_stout:
            backend_overview()
        stdout = fake_stout.getvalue()
        self.assertIn('Operational:', stdout)
        self.assertIn('Avg. T1:', stdout)
        self.assertIn('Num. Qubits:', stdout)

    @requires_qe_access
    def test_backend_monitor(self, qe_token, qe_url):
        """Test backend_monitor"""
        from qiskit import IBMQ  # pylint: disable: import-error
        IBMQ.enable_account(qe_token, qe_url)
        for back in IBMQ.backends():
            if not back.configuration().simulator:
                backend = back
                break
        with patch('sys.stdout', new=StringIO()) as fake_stout:
            backend_monitor(backend)

        stdout = fake_stout.getvalue()
        self.assertIn('Configuration', stdout)
        self.assertIn('Qubits [Name / Freq / T1 / T2 / U1 err / U2 err / U3 err / Readout err]',
                      stdout)
        self.assertIn('Multi-Qubit Gates [Name / Type / Gate Error]', stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
