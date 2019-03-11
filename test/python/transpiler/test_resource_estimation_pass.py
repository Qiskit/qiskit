# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""ResourceEstimation pass testing"""

import unittest

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import ResourceEstimation
from qiskit.test import QiskitTestCase


class TestResourceEstimationPass(QiskitTestCase):
    """ Tests for PropertySet methods. """

    def test_empty_dag(self):
        """ Empty DAG has 0 amount of operations """
        circuit = QuantumCircuit()
        dag = circuit_to_dag(circuit)

        pass_ = ResourceEstimation()
        _ = pass_.run(dag)

        self.assertEqual(pass_.property_set['size'], 0)
        self.assertEqual(pass_.property_set['depth'], 0)
        self.assertEqual(pass_.property_set['width'], 0)

    def test_count_h_and_cx(self):
        """ A dag with 8 operations """
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

        pass_ = ResourceEstimation()
        _ = pass_.run(dag)

        self.assertEqual(pass_.property_set['size'], 8)
        self.assertEqual(pass_.property_set['depth'], 7)
        self.assertEqual(pass_.property_set['width'], 2)

    def test_depth_one(self):
        """ A dag with operations in parallel and depth 1"""
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[1])
        dag = circuit_to_dag(circuit)

        pass_ = ResourceEstimation()
        _ = pass_.run(dag)

        self.assertEqual(pass_.property_set['size'], 2)
        self.assertEqual(pass_.property_set['depth'], 1)
        self.assertEqual(pass_.property_set['width'], 2)


if __name__ == '__main__':
    unittest.main()
