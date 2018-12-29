# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for qiskit.Result"""

import unittest

import qiskit
from ..common import QiskitTestCase, requires_qe_access


class TestQiskitResult(QiskitTestCase):
    """Test qiskit.Result API"""

    def setUp(self):
        qr = qiskit.QuantumRegister(1)
        cr = qiskit.ClassicalRegister(1)
        self._qc1 = qiskit.QuantumCircuit(qr, cr, name='qc1')
        self._qc2 = qiskit.QuantumCircuit(qr, cr, name='qc2')
        self._qc1.measure(qr[0], cr[0])
        self._qc2.x(qr[0])
        self._qc2.measure(qr[0], cr[0])

    @requires_qe_access
    def test_ibmq_result_fields(self, qe_token, qe_url):
        """Test components of a result from a remote simulator."""
        qiskit.IBMQ.enable_account(qe_token, qe_url)
        remote_backend = qiskit.IBMQ.get_backend(local=False, simulator=True)
        remote_result = qiskit.execute(self._qc1, remote_backend).result()
        self.assertEqual(remote_result.backend_name, remote_backend.name())
        self.assertIsInstance(remote_result.job_id, str)
        self.assertEqual(remote_result.status, 'COMPLETED')
        self.assertEqual(remote_result.results[0].status, 'DONE')


if __name__ == '__main__':
    unittest.main(verbosity=2)
