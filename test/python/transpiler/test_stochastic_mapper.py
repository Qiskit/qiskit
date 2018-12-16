# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the Stochastic Mapper pass"""

import unittest
from qiskit.transpiler.passes import StochasticMapper
from qiskit.mapper import Coupling
from qiskit.converters import circuit_to_dag
from qiskit import QuantumRegister, QuantumCircuit
from ..common import QiskitTestCase


class TestStochasticMapper(QiskitTestCase):
    """
    Tests the StochasticMapper pass.
    
    All of the tests use a fixed seed since the results
    may depend on it.
    """

    def test_trivial_case(self):
        """
         q0:--(+)-[U]-(+)-
               |       |
         q1:---.-------|--
                       |
         q2:-----------.--

         Coupling map: [1]--[0]--[2]
        """
        coupling = Coupling({0: [1, 2]})

        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[2])

        dag = circuit_to_dag(circuit)
        pass_ = StochasticMapper(coupling, None, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_trivial_in_same_layer(self):
        """
         q0:--(+)--
               |
         q1:---.---

         q2:--(+)--
               |
         q3:---.---

         Coupling map: [0]--[1]--[2]--[3]
        """
        coupling = Coupling({0: [1], 1: [2], 2: [3]})

        qr = QuantumRegister(4, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[0], qr[1])

        dag = circuit_to_dag(circuit)
        pass_ = StochasticMapper(coupling, None, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_permute_wires_1(self):
        """All of the test_permute_wires tests are derived
        from the basic mapper tests. In this case, the
        stochastic mapper handles a single
        layer by qubit label permutations so as not to
        introduce additional swap gates.
         q0:-------

         q1:--(+)--
               |
         q2:---.---

         Coupling map: [1]--[0]--[2]

         q0:-(+)--
              |
         q1:--.---

         q2:------

        """
        coupling = Coupling({0: [1, 2]})

        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.cx(qr[1], qr[0])

        pass_ = StochasticMapper(coupling, None, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_permute_wires_2(self):
        """
         qr0:---.---[H]--
                |
         qr1:---|--------
                |
         qr2:--(+)-------

         Coupling map: [0]--[1]--[2]

         qr0:---.--[H]--
                |
         qr1:--(+)------

         qr2:--------------
        """
        coupling = Coupling({1: [0, 2]})

        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.h(qr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.cx(qr[0], qr[1])
        expected.h(qr[0])

        pass_ = StochasticMapper(coupling, None, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_permute_wires_3(self):
        """
         qr0:--(+)---.--
                |    |
         qr1:---|----|--
                |    |
         qr2:---|----|--
                |    |
         qr3:---.---(+)-

         Coupling map: [0]--[1]--[2]--[3]
         For this seed,  we get the (1,2) edge.

         qr0:-----------

         qr1:---.---(+)-
                |    |
         qr2:--(+)---.--

         qr3:-----------
        """
        coupling = Coupling({0: [1], 1: [2], 2: [3]})

        qr = QuantumRegister(4, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[3], qr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.cx(qr[1], qr[2])
        expected.cx(qr[2], qr[1])

        pass_ = StochasticMapper(coupling, None, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_permute_wires_4(self):
        """No qubit label permutation occurs if the first
        layer has only single-qubit gates. This is suboptimal but is
        the current behavior.
         qr0:------(+)--
                    |
         qr1:-------|---
                    |
         qr2:-------|---
                    |
         qr3:--[H]--.---

         Coupling map: [0]--[1]--[2]--[3]

         qr0:------X---------
                   |
         qr1:------X-(+)-----
                      |
         qr2:------X--.------
                   |
         qr3:-[H]--X---------

        """
        coupling = Coupling({0: [1], 1: [2], 2: [3]})

        qr = QuantumRegister(4, 'q')
        circuit = QuantumCircuit(qr)
        circuit.h(qr[3])
        circuit.cx(qr[3], qr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.h(qr[3])
        expected.swap(qr[2], qr[3])
        expected.swap(qr[0], qr[1])
        expected.cx(qr[2], qr[1])

        pass_ = StochasticMapper(coupling, None, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_permute_wires_5(self):
        """
         qr0:--(+)------
                |
         qr1:---|-------
                |
         qr2:---|-------
                |
         qr3:---.--[H]--

         Coupling map: [0]--[1]--[2]--[3]
         For this seed, the mapper permutes these labels
         onto the (1,2) edge.

         qr0:------------

         qr1:---(+)------
                 |
         qr2:----.--[H]--

         qr3:------------

        """
        coupling = Coupling({0: [1], 1: [2], 2: [3]})

        qr = QuantumRegister(4, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[3], qr[0])
        circuit.h(qr[3])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.cx(qr[2], qr[1])
        expected.h(qr[2])

        pass_ = StochasticMapper(coupling, None, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_permute_wires_6(self):
        """
         qr0:--(+)-------.--
                |        |
         qr1:---|--------|--
                |
         qr2:---|--------|--
                |        |
         qr3:---.--[H]--(+)-

         Coupling map: [0]--[1]--[2]--[3]
         For this seed, the mapper permutes these labels
         onto the (1,2) edge.

         qr0:---------------------

         qr1:-------(+)-------.---
                     |        |
         qr2:--------.--[H]--(+)--

         qr3:---------------------

        """
        coupling = Coupling({0: [1], 1: [2], 2: [3]})

        qr = QuantumRegister(4, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[3], qr[0])
        circuit.h(qr[3])
        circuit.cx(qr[0], qr[3])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.cx(qr[2], qr[1])
        expected.h(qr[2])
        expected.cx(qr[1], qr[2])

        pass_ = StochasticMapper(coupling, None, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

if __name__ == '__main__':
    unittest.main()
