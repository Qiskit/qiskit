# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Commutation analysis and transformation pass testing"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import PropertySet
from qiskit.transpiler.passes import CommutationAnalysis
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase


class TestCommutationAnalysis(QiskitTestCase):
    """Test the Communttion pass."""

    def setUp(self):

        self.pass_ = CommutationAnalysis()
        self.pset = self.pass_.property_set = PropertySet()

    def test_commutation_set_property_is_created(self):
        """Test property is created"""
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        dag = circuit_to_dag(circuit)

        self.assertIsNone(self.pset['commutation_set'])
        self.pass_.run(dag)
        self.assertIsNotNone(self.pset['commutation_set'])

    def test_all_gates(self):
        """Test all gates on 1 and 2 qubits

        qr0:----[H]---[x]---[y]---[t]---[s]---[rz]---[u1]---[u2]---[u3]---.---.---.--
                                                                          |   |   |
        qr1:-------------------------------------------------------------(+)-(Y)--.--
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.x(qr[0])
        circuit.y(qr[0])
        circuit.t(qr[0])
        circuit.s(qr[0])
        circuit.rz(0.5, qr[0])
        circuit.u1(0.5, qr[0])
        circuit.u2(0.5, 0.6, qr[0])
        circuit.u3(0.5, 0.6, 0.7, qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cy(qr[0], qr[1])
        circuit.cz(qr[0], qr[1])
        dag = circuit_to_dag(circuit)

        self.pass_.run(dag)
        self.assertEqual(self.pset["commutation_set"]["qr[0]"], [[1], [5], [6], [7], [8, 9, 10, 11],
                                                                 [12], [13], [14], [15], [16], [2]])
        self.assertEqual(self.pset["commutation_set"]["qr[1]"], [[3], [14], [15], [16], [4]])

    def test_non_commutative_circuit(self):
        """A simple circuit where no gates commute

        qr0:---[H]---

        qr1:---[H]---

        qr2:---[H]---
        """
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        dag = circuit_to_dag(circuit)

        self.pass_.run(dag)
        self.assertEqual(self.pset["commutation_set"]["qr[0]"], [[1], [7], [2]])
        self.assertEqual(self.pset["commutation_set"]["qr[1]"], [[3], [8], [4]])
        self.assertEqual(self.pset["commutation_set"]["qr[2]"], [[5], [9], [6]])

    def test_non_commutative_circuit_2(self):
        """A simple circuit where no gates commute

        qr0:----.-------------
                |
        qr1:---(+)------.-----
                        |
        qr2:---[H]-----(+)----
        """
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[2])
        circuit.cx(qr[1], qr[2])
        dag = circuit_to_dag(circuit)

        self.pass_.run(dag)
        self.assertEqual(self.pset["commutation_set"]["qr[0]"], [[1], [7], [2]])
        self.assertEqual(self.pset["commutation_set"]["qr[1]"], [[3], [7], [9], [4]])
        self.assertEqual(self.pset["commutation_set"]["qr[2]"], [[5], [8], [9], [6]])

    def test_commutative_circuit(self):
        """A simple circuit where two CNOTs commute

        qr0:----.------------
                |
        qr1:---(+)-----(+)---
                        |
        qr2:---[H]------.----
        """

        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[2])
        circuit.cx(qr[2], qr[1])
        dag = circuit_to_dag(circuit)

        self.pass_.run(dag)
        self.assertEqual(self.pset["commutation_set"]["qr[0]"], [[1], [7], [2]])
        self.assertEqual(self.pset["commutation_set"]["qr[1]"], [[3], [7, 9], [4]])
        self.assertEqual(self.pset["commutation_set"]["qr[2]"], [[5], [8], [9], [6]])

    def test_commutative_circuit_2(self):
        """A simple circuit where a CNOT and a Z gate commute,
        and a CNOT and a CNOT commute

        qr0:----.-----[Z]-----
                |
        qr1:---(+)----(+)----
                       |
        qr2:---[H]-----.----
        """

        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.z(qr[0])
        circuit.h(qr[2])
        circuit.cx(qr[2], qr[1])
        dag = circuit_to_dag(circuit)

        self.pass_.run(dag)
        self.assertEqual(self.pset["commutation_set"]["qr[0]"], [[1], [7, 8], [2]])
        self.assertEqual(self.pset["commutation_set"]["qr[1]"], [[3], [7, 10], [4]])
        self.assertEqual(self.pset["commutation_set"]["qr[2]"], [[5], [9], [10], [6]])

    def test_commutative_circuit_3(self):
        """A simple circuit where multiple gates commute

        qr0:----.-----[Z]-----.----[z]-----
                |             |
        qr1:---(+)----(+)----(+)----.------
                       |            |
        qr2:---[H]-----.-----[x]---(+)-----
        """
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[2])
        circuit.z(qr[0])
        circuit.cx(qr[2], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.x(qr[2])
        circuit.z(qr[0])
        circuit.cx(qr[1], qr[2])

        dag = circuit_to_dag(circuit)

        self.pass_.run(dag)
        self.assertEqual(self.pset["commutation_set"]["qr[0]"], [[1], [7, 9, 11, 13], [2]])
        self.assertEqual(self.pset["commutation_set"]["qr[1]"], [[3], [7, 10, 11], [14], [4]])
        self.assertEqual(self.pset["commutation_set"]["qr[2]"], [[5], [8], [10], [12, 14], [6]])

    def test_jordan_wigner_type_circuit(self):
        """A Jordan-Wigner type circuit where consecutive CNOTs commute

        qr0:----.-------------------------------------------------------------.----
                |                                                             |
        qr1:---(+)----.-------------------------------------------------.----(+)---
                      |                                                 |
        qr2:---------(+)----.-------------------------------------.----(+)---------
                            |                                     |
        qr3:---------------(+)----.-------------------------.----(+)---------------
                                  |                         |
        qr4:---------------------(+)----.-------------.----(+)---------------------
                                        |             |
        qr5:---------------------------(+)----[z]----(+)---------------------------
        """
        qr = QuantumRegister(6, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[4], qr[5])
        circuit.z(qr[5])
        circuit.cx(qr[4], qr[5])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[0], qr[1])

        dag = circuit_to_dag(circuit)

        self.pass_.run(dag)
        self.assertEqual(self.pset["commutation_set"]["qr[0]"], [[1], [13, 23], [2]])
        self.assertEqual(self.pset["commutation_set"]["qr[1]"], [[3], [13], [14, 22], [23], [4]])
        self.assertEqual(self.pset["commutation_set"]["qr[2]"], [[5], [14], [15, 21], [22], [6]])
        self.assertEqual(self.pset["commutation_set"]["qr[3]"], [[7], [15], [16, 20], [21], [8]])
        self.assertEqual(self.pset["commutation_set"]["qr[4]"], [[9], [16], [17, 19], [20], [10]])
        self.assertEqual(self.pset["commutation_set"]["qr[5]"], [[11], [17], [18], [19], [12]])

    def test_all_commute_circuit(self):
        """Test circuit with that all commute"""
        qr = QuantumRegister(5, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[1])
        circuit.cx(qr[4], qr[3])
        circuit.cx(qr[2], qr[3])
        circuit.z(qr[0])
        circuit.z(qr[4])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[1])
        circuit.cx(qr[4], qr[3])
        circuit.cx(qr[2], qr[3])
        dag = circuit_to_dag(circuit)

        self.pass_.run(dag)
        self.assertEqual(self.pset["commutation_set"]["qr[0]"], [[1], [11, 15, 17], [2]])
        self.assertEqual(self.pset["commutation_set"]["qr[1]"], [[3], [11, 12, 17, 18], [4]])
        self.assertEqual(self.pset["commutation_set"]["qr[2]"], [[5], [12, 14, 18, 20], [6]])
        self.assertEqual(self.pset["commutation_set"]["qr[3]"], [[7], [13, 14, 19, 20], [8]])
        self.assertEqual(self.pset["commutation_set"]["qr[4]"], [[9], [13, 16, 19], [10]])


if __name__ == '__main__':
    unittest.main()
