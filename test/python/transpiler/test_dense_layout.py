# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the DenseLayout pass"""

import unittest

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.mapper import CouplingMap, Layout
from qiskit.transpiler.passes import DenseLayout
from qiskit.transpiler import TranspilerError
from qiskit.converters import circuit_to_dag
from ..common import QiskitTestCase


class TestDenseLayout(QiskitTestCase):
    """Tests the DenseLayout pass"""

    def setUp(self):
        self.cmap5 = [[1, 0], [2, 0], [2, 1], [3, 2], [3, 4], [4, 2]]
        
        self.cmap16 = [[1, 0], [1, 2], [2, 3], [3, 4], [3, 14], [5, 4], [6, 5],
                       [6, 7], [6, 11], [7, 10], [8, 7], [9, 8], [9, 10],
                       [11, 10], [12, 5], [12, 11], [12, 13], [13, 4],
                       [13, 14], [15, 0], [15, 2], [15, 14]]

        self.cmap20 = [[0, 1], [0, 5], [1, 0], [1, 2], [1, 6], [1, 7], [2, 1],
                       [2, 3], [2, 6], [3, 2], [3, 8], [3, 9], [4, 8], [4, 9],
                       [5, 0], [5, 6], [5, 10], [5, 11], [6, 1], [6, 2], [6, 5],
                       [6, 7], [6, 10], [6, 11], [7, 1], [7, 6], [7, 8], [7, 12],
                       [7, 13], [8, 3], [8, 4], [8, 7], [8, 9], [8, 12], [8, 13],
                       [9, 3], [9, 4], [9, 8], [10, 5], [10, 6], [10, 11], [10, 15],
                       [11, 5], [11, 6], [11, 10], [11, 12], [11, 16], [11, 17],
                       [12, 7], [12, 8], [12, 11], [12, 13], [12, 16], [13, 7],
                       [13, 8], [13, 12], [13, 14], [13, 18], [13, 19], [14, 13],
                       [14, 18], [14, 19], [15, 10], [15, 16], [16, 11], [16, 12],
                       [16, 15], [16, 17], [17, 11], [17, 16], [18, 13], [18, 14],
                       [19, 13], [19, 14]]

    def test_3q_circuit_5q_coupling(self):
        """Test finds trivial mapping when already mapped.
        """
        qr = QuantumRegister(5, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[2], qr[1])
        circuit.cx(qr[4], qr[2])

        dag = circuit_to_dag(circuit)
        pass_ = DenseLayout(CouplingMap(self.cmap5))
        pass_.property_set['is_direction_mapped'] = True
        pass_.run(dag)

        layout = pass_.property_set['layout']
        self.assertEqual(layout[qr[1]], 1)
        self.assertEqual(layout[qr[2]], 2)
        self.assertEqual(layout[qr[4]], 4)

    def test_3q_circuit_16q_coupling(self):
        """Test finds trivial mapping when already mapped.
        """
        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[0], qr[1])

        dag = circuit_to_dag(circuit)
        pass_ = DenseLayout(CouplingMap(self.cmap16))
        pass_.property_set['is_direction_mapped'] = True
        pass_.run(dag)

        layout = pass_.property_set['layout']
        self.assertEqual(layout[qr[0]], 0)
        self.assertEqual(layout[qr[1]], 1)
        self.assertEqual(layout[qr[2]], 2)

    def test_5q_circuit_20q_coupling(self):
        """Test finds trivial mapping when already mapped.
        """
        qr = QuantumRegister(5, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[3], qr[1])
        circuit.cx(qr[0], qr[2])

        dag = circuit_to_dag(circuit)
        pass_ = DenseLayout(CouplingMap(self.cmap20))
        pass_.property_set['is_direction_mapped'] = False
        pass_.run(dag)

        layout = pass_.property_set['layout']
        self.assertEqual(layout[qr[0]], 0)
        self.assertEqual(layout[qr[1]], 1)
        self.assertEqual(layout[qr[2]], 2)


if __name__ == '__main__':
    unittest.main()
