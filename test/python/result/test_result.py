# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for qiskit.Result"""

import unittest

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute
from qiskit import BasicAer
from qiskit.test import QiskitTestCase


class TestQiskitResult(QiskitTestCase):
    """Test qiskit.Result API"""

    def setUp(self):
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        self._qc1 = QuantumCircuit(qr, cr, name='qc1')
        self._qc2 = QuantumCircuit(qr, cr, name='qc2')
        self._qc1.measure(qr[0], cr[0])
        self._qc2.x(qr[0])
        self._qc2.measure(qr[0], cr[0])
        self.backend = BasicAer.get_backend('qasm_simulator')
        self._result1 = execute(self._qc1, self.backend).result()
        self._result2 = execute(self._qc2, self.backend).result()

    def test_extend_result(self):
        """Test extending a Result instance is possible."""
        result1, result2 = (self._result1, self._result2)
        counts1 = result1.get_counts(self._qc1.name)
        counts2 = result2.get_counts(self._qc2.name)
        result1 += result2  # extend a result
        self.assertEqual(
            [
                result1.get_counts(self._qc1.name),
                result2.get_counts(self._qc2.name)
            ],
            [counts1, counts2]
        )

    def test_combine_results(self):
        """Test combining results in a new Result instance is possible."""
        result1, result2 = (self._result1, self._result2)
        counts1 = result1.get_counts(self._qc1.name)
        counts2 = result2.get_counts(self._qc2.name)
        new_result = result1 + result2  # combine results
        self.assertEqual(
            [
                new_result.get_counts(self._qc1.name),
                new_result.get_counts(self._qc2.name)
            ],
            [counts1, counts2]
        )
        self.assertIsNot(new_result, result1)
        self.assertIsNot(new_result, result2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
