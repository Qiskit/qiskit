# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the FlexlayerSwap pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import FlexlayerSwap


class TestFlexlayerSwap(QiskitTestCase):
    """ Tests the FlexlayerSwap pass."""

    def test_trivial_case(self):
        """No need to have any swap, the CX are distance 1 to each other
         q0:--(+)-[U]-(+)-
               |       |
         q1:---.-------|--
                       |
         q2:-----------.--

         CouplingMap map: [1]--[0]--[2]
        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        qr = QuantumRegister(3, name='q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[2], qr[0])

        dag = circuit_to_dag(circuit)
        pass_ = FlexlayerSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_a_single_swap(self):
        """ Adding a swap
         q0:-------

         q1:---.---
               |
         q2:--(+)--

         CouplingMap map: [1]--[0]--[2]

         q0:--X--(+)--
              |   |
         q1:--|---.---
              |
         q2:--x-------

        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        qr = QuantumRegister(3, name='q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.swap(qr[0], qr[2])
        expected.cx(qr[1], qr[0])

        pass_ = FlexlayerSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_far_swap(self):
        """ A far swap that affects coming CXs.
         qr0:--(+)---.--
                |    |
         qr1:---|----|--
                |    |
         qr2:---|----|--
                |    |
         qr3:---.---(+)-

         CouplingMap map: [0]--[1]--[2]--[3]

         qr0:--X--------------
               |
         qr1:--X--X-----------
                  |
         qr2:-----X--(+)---.--
                      |    |
         qr3:---------.---(+)-

        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, name='q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[3], qr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.swap(qr[0], qr[1])
        expected.swap(qr[1], qr[2])
        expected.cx(qr[2], qr[3])
        expected.cx(qr[3], qr[2])

        pass_ = FlexlayerSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)


if __name__ == '__main__':
    unittest.main()
