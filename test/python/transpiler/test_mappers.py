# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Meta Tests for mappers"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit, BasicAer
from qiskit.transpiler import PassManager, transpile
from qiskit.transpiler.passes import BasicSwap, LookaheadSwap
from qiskit.mapper import CouplingMap

from ..common import QiskitTestCase

class CommonUtilities():

    def create_passmanager(self, coupling_map):
        return PassManager(self.swap_pass(CouplingMap(coupling_map)))

    def assertResult(self, result, expected):
        self.assertEqual(result, expected)

class CommonTestCases(CommonUtilities):

    def test_a_cx_to_map(self):
        """ A single CX needs to be remapped
         q0:-------

         q1:-[H]-(+)--
                  |
         q2:------.---

         CouplingMap map: [1]<-[0]->[2]
        """
        coupling_map = [[0, 1], [0, 2]]

        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.h(qr[1])
        circuit.cx(qr[1], qr[2])

        backend = BasicAer.get_backend('qasm_simulator')
        result = transpile(circuit, backend, coupling_map=coupling_map,
                             pass_manager=self.create_passmanager(coupling_map))
        self.assertResult(result, self.expected['a_cx_to_map'])

class TestsBasicMapper(CommonTestCases, QiskitTestCase):

    def setUp(self):
        self.swap_pass = BasicSwap
        qr = QuantumRegister(3, 'q')
        self.expected = {}
        self.expected["a_cx_to_map"] = QuantumCircuit(qr)
        self.expected["a_cx_to_map"].h(qr[1])
        self.expected["a_cx_to_map"].swap(qr[1], qr[0])
        self.expected["a_cx_to_map"].cx(qr[0], qr[2])

    def test_specific_input(self):
        """Some specific test"""
        self.assertTrue(True)

class TestsLookaheadSwap(CommonTestCases, QiskitTestCase):
    def setUp(self):
        self.swap_pass = LookaheadSwap
        qr = QuantumRegister(3, 'q')
        self.expected = {}
        self.expected["a_cx_to_map"] = QuantumCircuit(qr)
        self.expected["a_cx_to_map"].h(qr[1])
        self.expected["a_cx_to_map"].swap(qr[0], qr[1])
        self.expected["a_cx_to_map"].cx(qr[0], qr[2])

    def test_specific_input(self):
        """Some specific test"""
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
