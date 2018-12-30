# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the DenseLayout pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.mapper import CouplingMap
from qiskit.transpiler.passes import DenseLayout
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase


class TestDenseLayout(QiskitTestCase):
    """Tests the DenseLayout pass"""

    def setUp(self):
        """
        0  =  1   =  2   =  3     4

        ||    ||    ||     ||  X  ||

        5  =  6   =  7   =  8  =  9

        || X  ||    ||   X  ||

        10 =  11  =  12  =  13 =  14

        ||    ||  X         || X  ||

        15 =  16  =  17     18    19
        """
        self.cmap20 = [[0, 1], [0, 5], [1, 0], [1, 2], [1, 6], [2, 1],
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

    def test_5q_circuit_20q_coupling(self):
        """Test finds dense 5q corner in 20q coupling map.
        """
        qr = QuantumRegister(5, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[3], qr[1])
        circuit.cx(qr[0], qr[2])

        dag = circuit_to_dag(circuit)
        pass_ = DenseLayout(CouplingMap(self.cmap20))
        pass_.run(dag)

        layout = pass_.property_set['layout']
        self.assertEqual(layout[qr[0]], 5)
        self.assertEqual(layout[qr[1]], 0)
        self.assertEqual(layout[qr[2]], 6)
        self.assertEqual(layout[qr[3]], 10)
        self.assertEqual(layout[qr[4]], 11)

    def test_6q_circuit_20q_coupling(self):
        """Test finds dense 5q corner in 20q coupling map.
        """
        qr0 = QuantumRegister(3, 'q0')
        qr1 = QuantumRegister(3, 'q1')
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr0[0], qr1[2])
        circuit.cx(qr1[1], qr0[2])

        dag = circuit_to_dag(circuit)
        pass_ = DenseLayout(CouplingMap(self.cmap20))
        pass_.run(dag)

        layout = pass_.property_set['layout']
        self.assertEqual(layout[qr0[0]], 5)
        self.assertEqual(layout[qr0[1]], 0)
        self.assertEqual(layout[qr0[2]], 6)
        self.assertEqual(layout[qr1[0]], 10)
        self.assertEqual(layout[qr1[1]], 11)
        self.assertEqual(layout[qr1[2]], 1)


if __name__ == '__main__':
    unittest.main()
