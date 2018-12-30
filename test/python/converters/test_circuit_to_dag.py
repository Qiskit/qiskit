# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for the converters."""

import unittest

from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.test import QiskitTestCase


class TestCircuitToDag(QiskitTestCase):
    """Test Circuit to DAG."""

    def test_circuit_and_dag(self):
        """Check convert to dag and back"""
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        circuit_in = QuantumCircuit(qr, cr)
        circuit_in.h(qr[0])
        circuit_in.h(qr[1])
        circuit_in.measure(qr[0], cr[0])
        circuit_in.measure(qr[1], cr[1])
        circuit_in.x(qr[0]).c_if(cr, 0x3)
        circuit_in.measure(qr[0], cr[0])
        circuit_in.measure(qr[1], cr[1])
        circuit_in.measure(qr[2], cr[2])
        dag = circuit_to_dag(circuit_in)
        circuit_out = dag_to_circuit(dag)
        self.assertEqual(circuit_out, circuit_in)


if __name__ == '__main__':
    unittest.main(verbosity=2)
