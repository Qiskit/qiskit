# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for qiskit.Result"""

import unittest

import qiskit
from qiskit import BasicAer
from ..common import QiskitTestCase


class TestQiskitResult(QiskitTestCase):
    """Test qiskit.Result API"""

    def setUp(self):
        qr = qiskit.QuantumRegister(1)
        cr = qiskit.ClassicalRegister(1)
        self._qc1 = qiskit.QuantumCircuit(qr, cr, name='qc1')
        self._qc2 = qiskit.QuantumCircuit(qr, cr, name='qc2')
        self._qc1.measure(qr[0], cr[0])
        self.backend = BasicAer.get_backend('qasm_simulator')
        self._result1 = qiskit.execute(self._qc1, self.backend).result()

    def test_builtin_simulator_result_fields(self):
        """Test components of a result from a local simulator."""

        self.assertEqual('qasm_simulator', self._result1.backend_name)
        self.assertIsInstance(self._result1.job_id, str)
        self.assertEqual(self._result1.status, 'COMPLETED')
        self.assertEqual(self._result1.results[0].status, 'DONE')


if __name__ == '__main__':
    unittest.main(verbosity=2)
