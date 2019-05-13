# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the Check CNOT direction pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler.passes import CheckCXDirection
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase


class TestCheckCNotDirection(QiskitTestCase):
    """ Tests the CheckCXDirection pass with CX gates"""

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
        pass_ = CheckCXDirection(coupling)
        pass_.run(dag)
        self.assertTrue(pass_.property_set['is_direction_mapped'])

    def test_true_direction(self):
        """ Mapped is easy to check
         qr0:---.--[H]--.--
                |       |
         qr1:--(+)------|--
                        |
         qr2:----------(+)-

         CouplingMap map: [1]<-[0]->[2]
        """
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[2])
        coupling = CouplingMap([[0, 1], [0, 2]])
        dag = circuit_to_dag(circuit)

        pass_ = CheckCXDirection(coupling)
        pass_.run(dag)

        self.assertTrue(pass_.property_set['is_direction_mapped'])

    def test_true_direction_in_same_layer(self):
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
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])
        dag = circuit_to_dag(circuit)

        pass_ = CheckCXDirection(coupling)
        pass_.run(dag)

        self.assertTrue(pass_.property_set['is_direction_mapped'])

    def test_wrongly_mapped(self):
        """ Needs [0]-[1] in a [0]--[2]--[1]
         qr0:--(+)--
                |
         qr1:---.---

         CouplingMap map: [0]->[2]->[1]
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        coupling = CouplingMap([[0, 2], [2, 1]])
        dag = circuit_to_dag(circuit)

        pass_ = CheckCXDirection(coupling)
        pass_.run(dag)

        self.assertFalse(pass_.property_set['is_direction_mapped'])

    def test_true_direction_undirected(self):
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
        coupling = CouplingMap([[0, 1], [0, 2]])
        dag = circuit_to_dag(circuit)

        pass_ = CheckCXDirection(coupling)
        pass_.run(dag)

        self.assertFalse(pass_.property_set['is_direction_mapped'])

    def test_false_direction_in_same_layer_undirected(self):
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
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])
        dag = circuit_to_dag(circuit)

        pass_ = CheckCXDirection(coupling)
        pass_.run(dag)

        self.assertFalse(pass_.property_set['is_direction_mapped'])


class TestCheckCNotDirectionBarrier(QiskitTestCase):
    """ Tests the CheckCNotDirection pass with Swap gates"""

    def test_2q_barrier(self):
        """ A 2q barrier should be ignored
         qr0:--|--
               |
         qr1:--|--

         CouplingMap map: None
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.barrier(qr[0], qr[1])
        coupling = CouplingMap()
        dag = circuit_to_dag(circuit)

        pass_ = CheckCXDirection(coupling)
        pass_.run(dag)

        self.assertTrue(pass_.property_set['is_direction_mapped'])


if __name__ == '__main__':
    unittest.main()
