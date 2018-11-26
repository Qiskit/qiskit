# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the Check Map pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler.passes import CheckMap
from qiskit.mapper import Coupling
from qiskit.dagcircuit import DAGCircuit
from ..common import QiskitTestCase


class TestCheckMap(QiskitTestCase):
    """ Tests the CheckMap pass."""

    def test_trivial_map(self):
        """ Trivial map in a circuit without entanglement
         qr0:---[H]---

         qr1:---[H]---

         qr2:---[H]---

         Coupling map: None
        """
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        coupling = Coupling()
        dag = DAGCircuit.fromQuantumCircuit(circuit)
        pass_ = CheckMap(coupling)
        pass_.run(dag)
        self.assertTrue(pass_.property_set['is_mapped'])

    def test_true_map(self):
        """ Mapped is easy to check
         qr0:--(+)-[H]-(+)-
                |       |
         qr1:---.-------|--
                        |
         qr2:-----------.--

         Coupling map: [1]--[0]--[2]
        """
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[2])
        coupling = Coupling(couplingdict={0: [1, 2]})
        dag = DAGCircuit.fromQuantumCircuit(circuit)

        pass_ = CheckMap(coupling)
        pass_.run(dag)

        self.assertTrue(pass_.property_set['is_mapped'])

    def test_true_map_in_same_layer(self):
        """ Two CXs distance 1 to each other, in the same layer
         qr0:--(+)--
               |
         qr1:---.---

         qr2:--(+)--
               |
         qr3:---.---

         Coupling map: [0]--[1]--[2]--[3]
        """
        qr = QuantumRegister(4, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        coupling = Coupling(couplingdict={0: [1], 1: [2], 2: [3]})
        dag = DAGCircuit.fromQuantumCircuit(circuit)

        pass_ = CheckMap(coupling)
        pass_.run(dag)

        self.assertTrue(pass_.property_set['is_mapped'])

    def test_false_map(self):
        """ Needs [0]-[1] in a [0]--[2]--[1]
         qr0:--(+)--
                |
         qr1:---.---

         Coupling map: [0]--[2]--[1]
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        coupling = Coupling(couplingdict={0: [2], 2: [1]})
        dag = DAGCircuit.fromQuantumCircuit(circuit)

        pass_ = CheckMap(coupling)
        pass_.run(dag)

        self.assertFalse(pass_.property_set['is_mapped'])


if __name__ == '__main__':
    unittest.main()
