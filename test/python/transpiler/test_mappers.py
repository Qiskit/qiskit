# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Meta-test for Mappers"""

import sys
from qiskit.transpiler.passes import BasicMapper
from qiskit.mapper import Coupling
from qiskit.converters import circuit_to_dag
from qiskit import QuantumRegister, QuantumCircuit
from ..common import QiskitTestCase

mappers = [BasicMapper]

class SetUp():
    def case_1(self):
        """  qr0:--.-------(+)--.---
                   |        |   |
             qr1:--|--------|--(+)--
                   |        |
             qr2:-(+)--[H]--|-------
                            |
             qr3:-----------.-------

             Coupling map: [0]->[1]->[2]->[3]
        """
        self.kargs = {'coupling_map': Coupling({0: [1], 1: [2], 2: [3]})}

        qr = QuantumRegister(4, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[2], qr[0])
        circuit.h(qr[2])
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[1], qr[0])
        self.input_dag = circuit_to_dag(circuit)

class ExpectedBasicMapper(QiskitTestCase):
    def case_1(self):
        qr = QuantumRegister(4, 'qr')
        expected = QuantumCircuit(qr)
        expected.swap(qr[2], qr[1])
        expected.cx(qr[1], qr[0])
        expected.swap(qr[0], qr[1])
        expected.h(qr[0])
        expected.swap(qr[1], qr[2])
        expected.cx(qr[2], qr[3])
        expected.cx(qr[1], qr[2])
        self.output_dag = circuit_to_dag(expected)

def test_generator(mapperClass):
    def test(self):
        after = mapperClass(**self.kargs).run(self.input_dag)
        self.assertEqual(self.output_dag, after)
    return test


for mapper in mappers:
    ExpetedMapperClass = getattr(sys.modules[__name__], 'Expected'+mapper.__name__)
    cases = [ attribute for attribute in ExpetedMapperClass.__dict__.keys() if attribute[0] != '_']
    for a_case in cases:
        test_name = 'test_%s' % a_case
        Test = type(mapper.__name__, (QiskitTestCase,), {})
        getattr(SetUp, a_case)(Test)
        getattr(ExpetedMapperClass, a_case)(Test)
        test_case = test_generator(mapper)
        setattr(Test, test_name, test_case)