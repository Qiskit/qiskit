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

import sys
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock
from io import StringIO

import qiskit
from qiskit import providers
from qiskit.tools.monitor import backend_overview, backend_monitor
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeProvider
from qiskit.test.mock import FakeBackend
from qiskit.test.mock import FakeVigo


FAKE_PROV = FakeProvider()


class TestBackendOverview(QiskitTestCase):
    """Tools test case."""

    def _restore_ibmq(self):
        if not self.import_error:
            qiskit.IBMQ = self.ibmq_back
        else:
            del qiskit.IBMQ
        if self.prov_backup:
            providers.ibmq = self.prov_backup
        else:
            del providers.ibmq

    def setUp(self):
        super().setUp()
        ibmq_mock = MagicMock()
        ibmq_mock.IBMQBackend = FakeBackend
        sys.modules['qiskit.providers.ibmq'] = ibmq_mock

        import qiskit
        if hasattr(qiskit, 'IBMQ'):
            self.import_error = False
        else:
            self.import_error = True
            qiskit.IBMQ = None
        self.ibmq_back = qiskit.IBMQ
        IBMQ = FakeProvider()
        self.addCleanup(self._restore_ibmq)
        if hasattr(providers, 'ibmq'):
            self.prov_backup = providers.ibmq
        else:
            self.prov_backup = None
        providers.ibmq = MagicMock()

    @patch('qiskit.tools.monitor.overview.get_unique_backends',
           return_value=[FakeVigo()])
    def test_backend_overview(self, _):
        """Test backend_overview"""
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            backend_overview()
        stdout = fake_stdout.getvalue()
        self.assertIn('Operational:', stdout)
        self.assertIn('Avg. T1:', stdout)
        self.assertIn('Num. Qubits:', stdout)

    @patch('qiskit.tools.monitor.overview.get_unique_backends',
           return_value=[FakeVigo()])
    def test_backend_monitor(self, _):
        """Test backend_monitor"""
        for back in [FakeVigo()]:
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
