# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for qiskit.quantum_info.analyzation"""

import unittest

import qiskit
from qiskit import BasicAer
from qiskit.quantum_info.analyzation.average import average_data
from qiskit.test import QiskitTestCase


class TestAnalyzation(QiskitTestCase):
    """Test qiskit.Result API"""

    def test_average_data(self):
        """Test average_data."""
        qr = qiskit.QuantumRegister(2)
        cr = qiskit.ClassicalRegister(2)
        qc = qiskit.QuantumCircuit(qr, cr, name="qc")
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        shots = 10000
        backend = BasicAer.get_backend('qasm_simulator')
        result = qiskit.execute(qc, backend, shots=shots).result()
        counts = result.get_counts(qc)
        observable = {"00": 1, "11": 1, "01": -1, "10": -1}
        mean_zz = average_data(counts=counts, observable=observable)
        observable = {"00": 1, "11": -1, "01": 1, "10": -1}
        mean_zi = average_data(counts, observable)
        observable = {"00": 1, "11": -1, "01": -1, "10": 1}
        mean_iz = average_data(counts, observable)
        self.assertAlmostEqual(mean_zz, 1, places=1)
        self.assertAlmostEqual(mean_zi, 0, places=1)
        self.assertAlmostEqual(mean_iz, 0, places=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
