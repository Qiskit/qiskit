# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the Direction Mapper pass"""

import unittest

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.transpiler import MapperError
from qiskit.transpiler.passes import DirectionMapper
from qiskit.mapper import Coupling
from qiskit.converters import circuit_to_dag
from ..common import QiskitTestCase


class TestDirectionMapper(QiskitTestCase):
    """ Tests the DirectionMapper pass."""

    def test_no_cnots(self):
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
        dag = circuit_to_dag(circuit)

        pass_ = DirectionMapper(coupling)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_direction_error(self):
        """ The mapping cannot be fixed by direction mapper
         qr0:---------

         qr1:---(+)---
                 |
         qr2:----.----

         Coupling map: [2] <- [0] -> [1]
        """
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        coupling = Coupling({0: [2, 1]})
        dag = circuit_to_dag(circuit)

        pass_ = DirectionMapper(coupling)

        with self.assertRaises(MapperError):
            pass_.run(dag)

    def test_direction_correct(self):
        """ The CX is in the right direction
         qr0:---(+)---
                 |
         qr1:----.----

         Coupling map: [0] -> [1]
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        coupling = Coupling({0: [1]})
        dag = circuit_to_dag(circuit)

        pass_ = DirectionMapper(coupling)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_direction_flip(self):
        """ Flip a CX
         qr0:----.----
                 |
         qr1:---(+)---

         Coupling map: [0] -> [1]

         qr0:-[H]-(+)-[H]--
                   |
         qr1:-[H]--.--[H]--
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])
        coupling = Coupling({0: [1]})
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.h(qr[0])
        expected.h(qr[1])
        expected.cx(qr[0], qr[1])
        expected.h(qr[0])
        expected.h(qr[1])

        pass_ = DirectionMapper(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_flip_with_measure(self):
        """
         qr0: -(+)-[m]-
                |   |
         qr1: --.---|--
                    |
         cr0: ------.--

         Coupling map: [0] -> [1]

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
        coupling = Coupling({0: [1]})
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr, cr)
        expected.h(qr[0])
        expected.h(qr[1])
        expected.cx(qr[0], qr[1])
        expected.h(qr[0])
        expected.h(qr[1])
        expected.measure(qr[0], cr[0])

        pass_ = DirectionMapper(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)


if __name__ == '__main__':
    unittest.main()
