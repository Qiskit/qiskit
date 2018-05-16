# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring

"""Test for the DAGCircuit object"""

import unittest

from qiskit.dagcircuit import DAGCircuit
from .common import QiskitTestCase


class TestDagCircuit(QiskitTestCase):
    """QasmParser"""
    def setUp(self):
        self.QASM_FILE_PATH = self._get_resource_path('qasm/example.qasm')

    def test_create(self):
        qubit0 = ('qr', 0)
        qubit1 = ('qr', 1)
        clbit0 = ('cr', 0)
        clbit1 = ('cr', 1)
        condition = None
        dag = DAGCircuit()
        dag.add_basis_element('h', 1, number_classical=0, number_parameters=0)
        dag.add_basis_element('cx', 2)
        dag.add_basis_element('x', 1)
        dag.add_basis_element('measure', 1, number_classical=1,
                              number_parameters=0)
        dag.add_qreg('qr', 2)
        dag.add_creg('cr', 2)
        dag.apply_operation_back('h', [qubit0], [], [], condition)
        dag.apply_operation_back('cx', [qubit0, qubit1], [],
                                 [], condition)
        dag.apply_operation_back('measure', [qubit1], [clbit1], [], condition)
        dag.apply_operation_back('x', [qubit1], [], [], ('cr', 1))
        dag.apply_operation_back('measure', [qubit0], [clbit0], [], condition)
        dag.apply_operation_back('measure', [qubit1], [clbit1], [], condition)

    def test_get_named_nodes(self):
        dag = DAGCircuit()
        dag.add_basis_element('h', 1, number_classical=0, number_parameters=0)
        dag.add_basis_element('cx', 2)
        dag.add_qreg('q', 3)
        dag.apply_operation_back('cx', [('q', 0), ('q', 1)])
        dag.apply_operation_back('h', [('q', 0)])
        dag.apply_operation_back('cx', [('q', 2), ('q', 1)])
        dag.apply_operation_back('cx', [('q', 0), ('q', 2)])
        dag.apply_operation_back('h', [('q', 2)])

        named_nodes = list(dag.get_named_nodes('cx'))
        self.assertEqual(3, len(named_nodes))
        # Since the ordering is not assured, we sort to make it certain.
        # We have asserted here that lower node id implies that it was applied earlier.
        named_nodes = list(sorted(named_nodes))
        node_qargs = [dag.multi_graph.node[node_id]["qargs"] for node_id in named_nodes]
        self.assertEqual([
            [('q', 0), ('q', 1)],
            [('q', 2), ('q', 1)],
            [('q', 0), ('q', 2)]], node_qargs)


if __name__ == '__main__':
    unittest.main()
