# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the Check Map pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler.passes import CheckMap
from qiskit.mapper import CouplingMap
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase


class TestCheckMapCX(QiskitTestCase):
    """ Tests the CheckMap pass with CX gates"""

    def test_trivial_map(self):
        """ Trivial map in a circuit without entanglement
         qr0:---[H]---

         qr1:---[H]---

         qr2:---[H]---

         CouplingMap map: None
        """
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        coupling = CouplingMap()
        dag = circuit_to_dag(circuit)
        pass_ = CheckMap(coupling)
        pass_.run(dag)
        self.assertTrue(pass_.property_set['is_swap_mapped'])
        self.assertTrue(pass_.property_set['is_direction_mapped'])

    def test_true_map(self):
        """ Mapped is easy to check
         qr0:--(+)-[H]-(+)-
                |       |
         qr1:---.-------|--
                        |
         qr2:-----------.--

         CouplingMap map: [1]<-[0]->[2]
        """
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[2])
        coupling = CouplingMap([(0, 1), (0, 2)])
        dag = circuit_to_dag(circuit)

        pass_ = CheckMap(coupling)
        pass_.run(dag)

        self.assertTrue(pass_.property_set['is_swap_mapped'])
        self.assertTrue(pass_.property_set['is_direction_mapped'])

    def test_true_map_in_same_layer(self):
        """ Two CXs distance_qubits 1 to each other, in the same layer
         qr0:--(+)--
                |
         qr1:---.---

         qr2:--(+)--
                |
         qr3:---.---

         CouplingMap map: [0]->[1]->[2]->[3]
        """
        qr = QuantumRegister(4, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        coupling = CouplingMap([(0, 1), (1, 2), (2, 3)])
        dag = circuit_to_dag(circuit)

        pass_ = CheckMap(coupling)
        pass_.run(dag)

        self.assertTrue(pass_.property_set['is_swap_mapped'])
        self.assertTrue(pass_.property_set['is_direction_mapped'])

    def test_false_map(self):
        """ Needs [0]-[1] in a [0]--[2]--[1]
         qr0:--(+)--
                |
         qr1:---.---

         CouplingMap map: [0]->[2]->[1]
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        coupling = CouplingMap([(0, 2), (2, 1)])
        dag = circuit_to_dag(circuit)

        pass_ = CheckMap(coupling)
        pass_.run(dag)

        self.assertFalse(pass_.property_set['is_swap_mapped'])
        self.assertFalse(pass_.property_set['is_direction_mapped'])

    def test_true_map_undirected(self):
        """ Mapped but with wrong direction
         qr0:--(+)-[H]--.--
                |       |
         qr1:---.-------|--
                        |
         qr2:----------(+)-

         CouplingMap map: [1]<-[0]->[2]
        """
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.cx(qr[2], qr[0])
        coupling = CouplingMap([(0, 1), (0, 2)])
        dag = circuit_to_dag(circuit)

        pass_ = CheckMap(coupling)
        pass_.run(dag)

        self.assertTrue(pass_.property_set['is_swap_mapped'])
        self.assertFalse(pass_.property_set['is_direction_mapped'])

    def test_true_map_in_same_layer_undirected(self):
        """ Two CXs in the same layer, but one is wrongly directed
         qr0:--(+)--
                |
         qr1:---.---

         qr2:---.---
                |
         qr3:--(+)--

         CouplingMap map: [0]->[1]->[2]->[3]
        """
        qr = QuantumRegister(4, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[3], qr[2])
        coupling = CouplingMap([(0, 1), (1, 2), (2, 3)])
        dag = circuit_to_dag(circuit)

        pass_ = CheckMap(coupling)
        pass_.run(dag)

        self.assertTrue(pass_.property_set['is_swap_mapped'])
        self.assertFalse(pass_.property_set['is_direction_mapped'])


class TestCheckMapCZ(QiskitTestCase):
    """ Tests the CheckMap pass with CZ gates"""

    def test_true_map(self):
        """ Mapped is easy to check
         qr0:--(Z)-[H]-(Z)-
                |       |
         qr1:---.-------|--
                        |
         qr2:-----------.--

         CouplingMap map: [1]<-[0]->[2]
        """
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cz(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.cz(qr[0], qr[2])
        coupling = CouplingMap([(0, 1), (0, 2)])
        dag = circuit_to_dag(circuit)

        pass_ = CheckMap(coupling)
        pass_.run(dag)

        self.assertTrue(pass_.property_set['is_swap_mapped'])
        self.assertTrue(pass_.property_set['is_direction_mapped'])

    def test_false_map(self):
        """ Needs [0]-[1] in a [0]--[2]--[1]
         qr0:--(Z)--
                |
         qr1:---.---

         CouplingMap map: [0]->[2]->[1]
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cz(qr[0], qr[1])
        coupling = CouplingMap([(0, 2), (2, 1)])
        dag = circuit_to_dag(circuit)

        pass_ = CheckMap(coupling)
        pass_.run(dag)

        self.assertFalse(pass_.property_set['is_swap_mapped'])
        self.assertFalse(pass_.property_set['is_direction_mapped'])

    def test_true_map_undirected(self):
        """ Mapped but with wrong direction
         qr0:--(Z)-[H]--.--
                |       |
         qr1:---.-------|--
                        |
         qr2:----------(Z)-

         CouplingMap map: [1]<-[0]->[2]
        """
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cz(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.cz(qr[2], qr[0])
        coupling = CouplingMap([(0, 1), (0, 2)])
        dag = circuit_to_dag(circuit)

        pass_ = CheckMap(coupling)
        pass_.run(dag)

        self.assertTrue(pass_.property_set['is_swap_mapped'])
        self.assertFalse(pass_.property_set['is_direction_mapped'])


class TestCheckMapSwap(QiskitTestCase):
    """ Tests the CheckMap pass with Swap gates"""

    def test_true_map(self):
        """ Mapped is easy to check

         qr0:--X-[H]-X--
               |     |
         qr1:--X-----|--
                     |
         qr2:--------X--

         CouplingMap map: [1]<-[0]->[2]
        """
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.swap(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.swap(qr[0], qr[2])
        coupling = CouplingMap([(0, 1), (0, 2)])
        dag = circuit_to_dag(circuit)

        pass_ = CheckMap(coupling)
        pass_.run(dag)

        self.assertTrue(pass_.property_set['is_swap_mapped'])
        self.assertFalse(pass_.property_set['is_direction_mapped'])

    def test_true_map_symmetric(self):
        """ Mapped and directed, because coupling map fully connected

         qr0:--X-[H]-
               |
         qr1:--X-----

         CouplingMap map: [0]<->[1]
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.swap(qr[0], qr[1])
        circuit.h(qr[0])
        coupling = CouplingMap([(0, 1), (1, 0)])
        dag = circuit_to_dag(circuit)

        pass_ = CheckMap(coupling)
        pass_.run(dag)

        self.assertTrue(pass_.property_set['is_swap_mapped'])
        self.assertTrue(pass_.property_set['is_direction_mapped'])

    def test_true_map_in_same_layer(self):
        """ Two SWAPs distance_qubits 1 to each other, in the same layer
         qr0:--X--
               |
         qr1:--X--

         qr2:--X--
               |
         qr3:--X--

         CouplingMap map: [0]->[1]->[2]->[3]
        """
        qr = QuantumRegister(4, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.swap(qr[0], qr[1])
        circuit.swap(qr[2], qr[3])
        coupling = CouplingMap([(0, 1), (1, 2), (2, 3)])
        dag = circuit_to_dag(circuit)

        pass_ = CheckMap(coupling)
        pass_.run(dag)

        self.assertTrue(pass_.property_set['is_swap_mapped'])
        self.assertFalse(pass_.property_set['is_direction_mapped'])

    def test_false_map(self):
        """ Needs [0]-[1] in a [0]--[2]--[1]
         qr0:--X--
               |
         qr1:--X--

         CouplingMap map: [0]->[2]->[1]
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.swap(qr[0], qr[1])
        coupling = CouplingMap([(0, 2), (2, 1)])
        dag = circuit_to_dag(circuit)

        pass_ = CheckMap(coupling)
        pass_.run(dag)

        self.assertFalse(pass_.property_set['is_swap_mapped'])
        self.assertFalse(pass_.property_set['is_direction_mapped'])


if __name__ == '__main__':
    unittest.main()
