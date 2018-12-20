# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Meta Tests for mappers

Regenerate ground truth by running this on the root directory:
> python -m  test.python.transpiler.test_mappers

"""

# pylint: disable=redefined-builtin

import unittest
import pickle

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, BasicAer, compile
from qiskit.transpiler import PassManager, transpile
from qiskit.transpiler.passes import BasicSwap, LookaheadSwap, StochasticSwap
from qiskit.mapper import CouplingMap

from ..common import QiskitTestCase


class CommonUtilities():
    """ Some utilities for meta testing."""
    regenerate_expected = False
    seed = 42

    def create_passmanager(self, coupling_map):
        ''' Returns a PassManager using self.pass_class and coupling_map '''
        return PassManager(self.pass_class(CouplingMap(coupling_map)))

    def create_backend(self):
        ''' Returns a Backend.'''
        return BasicAer.get_backend('qasm_simulator')

    def generate_expected(self, result, filename):
        """
        Checks if result.get_count matches self.count by running in a backend
        (self.create_backend()). That's saved in a pickle in filename.

        Args:
            result (DAGCircuit): The DAGCircuit to compile and run.
            filename (string): Where the pickle is saved.
        """
        sim_backend = self.create_backend()
        qobj = compile(result, sim_backend, seed=self.seed)
        job = sim_backend.run(qobj)
        self.assertDictAlmostEqual(self.count, job.result().get_counts(), delta=self.delta)

        with open(filename, "wb") as output_file:
            pickle.dump(result, output_file)

    def assertResult(self, result, testname):
        ''' Fetches the pickle in testname file and compares it with result'''
        filename = self._get_resource_path('pickles/%s_%s.pickle' % (type(self).__name__, testname))

        if self.regenerate_expected:
            # Run result in backend to test that is valid.
            self.generate_expected(result, filename)

        with open(filename, "rb") as input_file:
            expected = pickle.load(input_file)

        self.assertEqual(result, expected)


class CommonTestCases(CommonUtilities):
    """ The tests here will be run in several mappers."""

    def test_a_cx_to_map(self):
        """ A single CX needs to be remapped
         q0:----------m-----
                      |
         q1:-[H]-(+)--|-m---
                  |   | |
         q2:------.---|-|-m-
                      | | |
         c0:----------.-|-|-
         c1:------------.-|-
         c2:--------------.-

         CouplingMap map: [1]<-[0]->[2]

        expected count: '000': 50%
                        '110': 50%
        """
        self.count = {'000': 512, '110': 512}
        self.delta = 5
        coupling_map = [[0, 1], [0, 2]]

        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(3, 'c')
        circuit = QuantumCircuit(qr, cr, name='a_cx_to_map')
        circuit.h(qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.measure(qr, cr)

        result = transpile(circuit, self.create_backend(), coupling_map=coupling_map,
                           pass_manager=self.create_passmanager(coupling_map))
        self.assertResult(result, circuit.name)

    def test_handle_measurement(self):
        """ Handle measurement correctly
         q0:--.-----(+)-m-------
              |      |  |
         q1:-(+)-(+)-|--|-m-----
                  |  |  | |
         q2:------|--|--|-|-m---
                  |  |  | | |
         q3:-[H]--.--.--|-|-|-m-
                        | | | |
         c0:------------.-|-|-|-
         c1:--------------.-|-|-
         c2:----------------.-|-
         c3:------------------.-

         CouplingMap map: [0]->[1]->[2]->[3]

        expected count: '0000': 50%
                        '1011': 50%
        """
        self.count = {'0000': 512, '1011': 512}
        self.delta = 5
        coupling_map = [[0, 1], [1, 2], [2, 3]]

        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr, name='handle_measurement')
        circuit.h(qr[3])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[3], qr[1])
        circuit.cx(qr[3], qr[0])
        circuit.measure(qr, cr)

        result = transpile(circuit, self.create_backend(), coupling_map=coupling_map,
                           pass_manager=self.create_passmanager(coupling_map))
        self.assertResult(result, circuit.name)


class TestsBasicMapper(CommonTestCases, QiskitTestCase):
    """ Test CommonTestCases using BasicSwap """
    pass_class = BasicSwap


class TestsLookaheadSwap(CommonTestCases, QiskitTestCase):
    """ Test CommonTestCases using LookaheadSwap """
    pass_class = LookaheadSwap

class TestsStochasticSwap(CommonTestCases, QiskitTestCase):
    """ Test CommonTestCases using StochasticSwap """
    pass_class = StochasticSwap

if __name__ == '__main__':
    CommonUtilities.regenerate_expected = True
    unittest.main()
