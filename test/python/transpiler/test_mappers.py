# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Meta Tests for mappers

Regenerate ground truth by running this on the root directory:
python -m  test.python.transpiler.test_mappers

"""

import unittest
import pickle

from qiskit import QuantumRegister, QuantumCircuit, BasicAer
from qiskit.transpiler import PassManager, transpile
from qiskit.transpiler.passes import BasicSwap, LookaheadSwap
from qiskit.mapper import CouplingMap

from ..common import QiskitTestCase

class CommonUtilities():
    regenerate_expected = False

    def create_passmanager(self, coupling_map):
        return PassManager(self.pass_class(CouplingMap(coupling_map)))

    def create_backend(self):
        return BasicAer.get_backend('qasm_simulator')

    def assertResult(self, result, test_name):
        filename = "%s_%s.pickle" % (type(self).__name__, test_name)
        if self.regenerate_expected:
            # Run result in backend to test that is valid.
            with open(filename, "wb") as output_file:
                pickle.dump(result, output_file)
        else:
            with open(filename, "rb") as input_file:
                expected=pickle.load(input_file)
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

        result = transpile(circuit, self.create_backend(), coupling_map=coupling_map,
                             pass_manager=self.create_passmanager(coupling_map))
        self.assertResult(result, 'a_cx_to_map')

class TestsBasicMapper(CommonTestCases, QiskitTestCase):
    pass_class = BasicSwap

    def test_specific_input(self):
        """Some specific test"""
        self.assertTrue(True)

class TestsLookaheadSwap(CommonTestCases, QiskitTestCase):
    pass_class = LookaheadSwap

    def test_specific_input(self):
        """Some specific test"""
        self.assertTrue(True)

if __name__ == '__main__':
    CommonUtilities.regenerate_expected = True
    unittest.main()
