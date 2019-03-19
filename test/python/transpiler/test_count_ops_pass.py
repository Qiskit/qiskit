# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Depth pass testing"""

import unittest

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import CountOps
from qiskit.test import QiskitTestCase


class TestCountOpsPass(QiskitTestCase):
    """ Tests for CountOps analysis methods. """

    def test_empty_dag(self):
        """ Empty DAG has empty counts."""
        circuit = QuantumCircuit()
        dag = circuit_to_dag(circuit)

        pass_ = CountOps()
        _ = pass_.run(dag)

        self.assertDictEqual(pass_.property_set['count_ops'], {})

    def test_just_qubits(self):
        """ A dag with 8 operations (6 CXs and 2 Hs)"""
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[1], qr[0])
        dag = circuit_to_dag(circuit)

        pass_ = CountOps()
        _ = pass_.run(dag)

        self.assertDictEqual(pass_.property_set['count_ops'], {'cx': 6, 'h': 2})


if __name__ == '__main__':
    unittest.main()
