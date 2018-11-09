# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the Swap Mapper pass"""

import unittest

from qiskit.transpiler.passes import SwapMapper
from qiskit.mapper import Coupling
from qiskit.dagcircuit import DAGCircuit
from qiskit import QuantumRegister
from ..common import QiskitTestCase

class TestSwapMapper(QiskitTestCase):
    def setUp(self):
        coupling_dict = {0: [1, 2]}
        self.coupling = Coupling(coupling_dict)

    def test_trivial_case(self):
        pass_ = SwapMapper(self.coupling)
        qreg = QuantumRegister(3, 'q')
        dag = DAGCircuit()
        dag.add_basis_element('U', 1, number_classical=0, number_parameters=0)
        dag.add_basis_element('CX', 2)
        dag.add_qreg(qreg)
        dag.apply_operation_back('CX', [('q', 0), ('q', 1)])
        dag.apply_operation_back('U', [('q', 0)])
        dag.apply_operation_back('CX', [('q', 0), ('q', 2)])
        dag.apply_operation_back('U', [('q', 0)])
        before = dag.qasm()
        after_dag = pass_.run(dag)
        self.assertEqual(before, after_dag.qasm())

if __name__ == '__main__':
    unittest.main()
