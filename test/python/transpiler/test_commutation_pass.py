# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Commutation analysis and transformation pass testing"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import PropertySet
from qiskit.transpiler.passes import CommutationAnalysis, CommutationTransformation
from ..common import QiskitTestCase

class TestCommutationPass(QiskitTestCase):
    
    def setUp(self):

        self.pass_ = CommutationAnalysis()
        self.pset = self.pass_.property_set = PropertySet()
    
    def test_commutation_set_property_is_created(self):
        """ The property set does not have a property called "fixed_point" and it is created after
        the  first run of the pass. """
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        dag = DAGCircuit.fromQuantumCircuit(circuit)

        self.assertIsNone(self.pset['commutation_set'])
        self.pass_.run(dag)
        self.assertIsNotNone(self.pset['commutation_set'])
    
    def test_non_commutative_circuit(self):
        """ A simple circuit that no gates commute
        qr0:---[H]---

        qr1:---[H]---

        qr2:---[H]---
        """
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        dag = DAGCircuit.fromQuantumCircuit(circuit)

        self.pass_.run(dag)
        self.assertEqual(self.pset["commutation_set"]["qr[0]"], [[1],[7],[2]])
        self.assertEqual(self.pset["commutation_set"]["qr[1]"], [[3],[8],[4]])
        self.assertEqual(self.pset["commutation_set"]["qr[2]"], [[5],[9],[6]])
         
    def test_non_commutative_circuit_2(self):

        """ A simple circuit that no gates commute
        qr0:---[Ctrl]---

        qr1:---[NOT]---[Ctrl]---

        qr2:---[H]------[NOT]----
        """
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[2])
        circuit.cx(qr[1], qr[2])
        dag = DAGCircuit.fromQuantumCircuit(circuit)

        self.pass_.run(dag)
        self.assertEqual(self.pset["commutation_set"]["qr[0]"], [[1],[7],[2]])
        self.assertEqual(self.pset["commutation_set"]["qr[1]"], [[3],[7],[9],[4]])
        self.assertEqual(self.pset["commutation_set"]["qr[2]"], [[5],[8],[9],[6]])

    def test_commutative_circuit(self):

        """ A simple circuit that two CNOTs commute
        qr0:---[Ctrl]---

        qr1:---[NOT]----[NOT]---

        qr2:---[H]------[Ctrl]----
        """

        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[2])
        circuit.cx(qr[2], qr[1])
        dag = DAGCircuit.fromQuantumCircuit(circuit)

        self.pass_.run(dag)
        self.assertEqual(self.pset["commutation_set"]["qr[0]"], [[1],[7],[2]])
        self.assertEqual(self.pset["commutation_set"]["qr[1]"], [[3],[7, 9],[4]])
        self.assertEqual(self.pset["commutation_set"]["qr[2]"], [[5],[8],[9],[6]])


    def test_commutative_circuit_2(self):

        """ A simple circuit that a CNOT and a Z gate commute, and a CNOT and a CNOT commute
        qr0:---[Ctrl]-----[Z]-----

        qr1:---[NOT]----[NOT]----

        qr2:---[H]------[Ctrl]----
        """

        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.z(qr[0])
        circuit.h(qr[2])
        circuit.cx(qr[2], qr[1])
        dag = DAGCircuit.fromQuantumCircuit(circuit)

        self.pass_.run(dag)
        self.assertEqual(self.pset["commutation_set"]["qr[0]"], [[1],[7, 8],[2]])
        self.assertEqual(self.pset["commutation_set"]["qr[1]"], [[3],[7, 10],[4]])
        self.assertEqual(self.pset["commutation_set"]["qr[2]"], [[5],[9],[10],[6]])

    def test_commutative_circuit_3(self):

        """ A simple circuit that multiple gates commute
        qr0:---[Ctrl]-----[Z]----[Ctrl]---[z]-------

        qr1:---[NOT]----[NOT]----[NOT]----[Ctrl]----

        qr2:---[H]------[Ctrl]---[x]------[NOT]-------
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

        dag = DAGCircuit.fromQuantumCircuit(circuit)

        self.pass_.run(dag)
        self.assertEqual(self.pset["commutation_set"]["qr[0]"], [[1],[7, 9, 11, 13],[2]])
        self.assertEqual(self.pset["commutation_set"]["qr[1]"], [[3],[7, 10, 11],[14],[4]])
        self.assertEqual(self.pset["commutation_set"]["qr[2]"], [[5],[8],[10],[12,14],[6]])

    def test_jordan_wigner_type_circuit(self):
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

        dag = DAGCircuit.fromQuantumCircuit(circuit)

        self.pass_.run(dag)
        self.assertEqual(self.pset["commutation_set"]["qr[0]"], [[1],[13, 23],[2]])
        self.assertEqual(self.pset["commutation_set"]["qr[1]"], [[3],[13],[14,22],[23],[4]])
        self.assertEqual(self.pset["commutation_set"]["qr[2]"], [[5],[14],[15,21],[22],[6]])
        self.assertEqual(self.pset["commutation_set"]["qr[3]"], [[7],[15],[16,20],[21],[8]])
        self.assertEqual(self.pset["commutation_set"]["qr[4]"], [[9],[16],[17,19],[20],[10]])
        self.assertEqual(self.pset["commutation_set"]["qr[5]"], [[11],[17],[18],[19],[12]])

    def test_all_commute_circuit(self):
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
        dag = DAGCircuit.fromQuantumCircuit(circuit)

        self.pass_.run(dag)
        self.assertEqual(self.pset["commutation_set"]["qr[0]"], [[1],[11, 15, 17],[2]])
        self.assertEqual(self.pset["commutation_set"]["qr[1]"], [[3],[11,12, 17, 18],[4]])
        self.assertEqual(self.pset["commutation_set"]["qr[2]"], [[5],[12, 14, 18, 20], [6]])
        self.assertEqual(self.pset["commutation_set"]["qr[3]"], [[7],[13, 14, 19,20],[8]])
        self.assertEqual(self.pset["commutation_set"]["qr[4]"], [[9],[13, 16, 19],[10]])

if __name__ == '__main__':
    unittest.main()
