# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring

"""Test for the DAGCircuit object"""

import unittest

from qiskit.dagcircuit import DAGCircuit
from .common import QiskitTestCase


class TestDagCircuit(QiskitTestCase):
    """Testing the dag circuit representation"""
    def test_create(self):
        qubit0 = ('qr', 0)
        qubit1 = ('qr', 1)
        clbit0 = ('cr', 0)
        clbit1 = ('cr', 1)
        condition = ('cr', 3)
        dag = DAGCircuit()
        dag.add_basis_element('h', 1, number_classical=0, number_parameters=0)
        dag.add_basis_element('cx', 2)
        dag.add_basis_element('x', 1)
        dag.add_basis_element('measure', 1, number_classical=1, number_parameters=0)
        dag.add_qreg('qr', 2)
        dag.add_creg('cr', 2)
        dag.apply_operation_back('h', [qubit0], [], [], condition=None)
        dag.apply_operation_back('cx', [qubit0, qubit1], [], [], condition=None)
        dag.apply_operation_back('measure', [qubit1], [clbit1], [], condition=None)
        dag.apply_operation_back('x', [qubit1], [], [], condition=condition)
        dag.apply_operation_back('measure', [qubit0], [clbit0], [], condition=None)
        dag.apply_operation_back('measure', [qubit1], [clbit1], [], condition=None)
        self.assertEqual(len(dag.multi_graph.nodes), 14)
        self.assertEqual(len(dag.multi_graph.edges), 16)

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

        # The ordering is not assured, so we only compare the output (unordered) sets.
        # We use tuples because lists aren't hashable.
        named_nodes = dag.get_named_nodes('cx')
        node_qargs = {tuple(dag.multi_graph.node[node_id]["qargs"]) for node_id in named_nodes}
        expected_gates = {
            (('q', 0), ('q', 1)),
            (('q', 2), ('q', 1)),
            (('q', 0), ('q', 2))}
        self.assertEqual(expected_gates, node_qargs)

    def test_layers_basic(self):
        qubit0 = ('qr', 0)
        qubit1 = ('qr', 1)
        clbit0 = ('cr', 0)
        clbit1 = ('cr', 1)
        condition = ('cr', 3)
        dag = DAGCircuit()
        dag.add_basis_element('h', 1, number_classical=0, number_parameters=0)
        dag.add_basis_element('cx', 2)
        dag.add_basis_element('x', 1)
        dag.add_basis_element('measure', 1, number_classical=1, number_parameters=0)
        dag.add_qreg('qr', 2)
        dag.add_creg('cr', 2)
        dag.apply_operation_back('h', [qubit0], [], [], condition=None)
        dag.apply_operation_back('cx', [qubit0, qubit1], [], [], condition=None)
        dag.apply_operation_back('measure', [qubit1], [clbit1], [], condition=None)
        dag.apply_operation_back('x', [qubit1], [], [], condition=condition)
        dag.apply_operation_back('measure', [qubit0], [clbit0], [], condition=None)
        dag.apply_operation_back('measure', [qubit1], [clbit1], [], condition=None)

        layers = list(dag.layers())
        self.assertEqual(5, len(layers))

        name_layers = [
            [node[1]["name"]
             for node in layer["graph"].multi_graph.nodes(data=True)
             if node[1]["type"] == "op"] for layer in layers]

        self.assertEqual([
            ['h'],
            ['cx'],
            ['measure'],
            ['x'],
            ['measure', 'measure']
            ], name_layers)


if __name__ == '__main__':
    unittest.main()
