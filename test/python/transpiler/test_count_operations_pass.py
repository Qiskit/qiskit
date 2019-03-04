# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""CountOperations pass testing"""

import unittest

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import CountOperations
from qiskit.test import QiskitTestCase


class TestCountGatesPass(QiskitTestCase):
    """ Tests for PropertySet methods. """

    def test_empty_dag(self):
        """ When pass_that_updates_the_property is not passed, there are no requirements. """
        self.assertEqual(len(self.pass_.requires), 0)

    def test_count_h_and_cx(self):
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[1], qr[0])
        dag = circuit_to_dag(circuit)

        pass_ = CountOperations()
        _ = pass_.run(dag)
        self.assertEqual(pass_.property_set['amount_of_operations'], 8)

if __name__ == '__main__':
    unittest.main()
