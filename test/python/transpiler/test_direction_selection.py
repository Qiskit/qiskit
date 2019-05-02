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

"""Test the CX Direction  pass"""
import unittest
from math import pi

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.transpiler import TranspilerError
from qiskit.mapper import CouplingMap
from qiskit.transpiler.passes import CXDirection
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase


class TestCXDirection(QiskitTestCase):
    """ Tests the CXDirection pass."""

    def test_no_cnots(self):
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

        pass_ = CXDirection(coupling)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_direction_error(self):
        """ The mapping cannot be fixed by direction mapper
         qr0:---------

         qr1:---(+)---
                 |
         qr2:----.----

         CouplingMap map: [2] <- [0] -> [1]
        """
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        coupling = CouplingMap([[0, 1], [0, 2]])
        dag = circuit_to_dag(circuit)

        pass_ = CXDirection(coupling)

        with self.assertRaises(TranspilerError):
            pass_.run(dag)

    def test_direction_correct(self):
        """ The CX is in the right direction
         qr0:---(+)---
                 |
         qr1:----.----

         CouplingMap map: [0] -> [1]
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        coupling = CouplingMap([[0, 1]])
        dag = circuit_to_dag(circuit)

        pass_ = CXDirection(coupling)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_direction_flip(self):
        """ Flip a CX
         qr0:----.----
                 |
         qr1:---(+)---

         CouplingMap map: [0] -> [1]

         qr0:-[H]-(+)-[H]--
                   |
         qr1:-[H]--.--[H]--
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])
        coupling = CouplingMap([[0, 1]])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.u2(0, pi, qr[0])
        expected.u2(0, pi, qr[1])
        expected.cx(qr[0], qr[1])
        expected.u2(0, pi, qr[0])
        expected.u2(0, pi, qr[1])

        pass_ = CXDirection(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_flip_with_measure(self):
        """
         qr0: -(+)-[m]-
                |   |
         qr1: --.---|--
                    |
         cr0: ------.--

         CouplingMap map: [0] -> [1]

         qr0: -[H]--.--[H]-[m]-
                    |       |
         qr1: -[H]-(+)-[H]--|--
                            |
         cr0: --------------.--
        """
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(1, 'cr')

        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[1], qr[0])
        circuit.measure(qr[0], cr[0])
        coupling = CouplingMap([[0, 1]])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.u2(0, pi, qr[0])
        expected.u2(0, pi, qr[1])
        expected.cx(qr[0], qr[1])
        expected.u2(0, pi, qr[0])
        expected.u2(0, pi, qr[1])
        expected.measure(qr[0], cr[0])

        pass_ = CXDirection(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)


if __name__ == '__main__':
    unittest.main()
