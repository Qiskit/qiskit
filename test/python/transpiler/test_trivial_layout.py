# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the TrivialLayout pass"""

import unittest

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.mapper import CouplingMap
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler import TranspilerError
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase


class TestDenseLayout(QiskitTestCase):
    """Tests the TrivialLayout pass"""

    def setUp(self):
        self.cmap5 = CouplingMap([[1, 0], [2, 0], [2, 1], [3, 2], [3, 4], [4, 2]])

        self.cmap16 = CouplingMap([[1, 0], [1, 2], [2, 3], [3, 4], [3, 14], [5, 4],
                                   [6, 5], [6, 7], [6, 11], [7, 10], [8, 7], [9, 8],
                                   [9, 10], [11, 10], [12, 5], [12, 11], [12, 13],
                                   [13, 4], [13, 14], [15, 0], [15, 2], [15, 14]])

    def test_3q_circuit_5q_coupling(self):
        """Test finds trivial layout for 3q circuit on 5q device.
        """
        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[1], qr[2])

        dag = circuit_to_dag(circuit)
        pass_ = TrivialLayout(self.cmap5)
        pass_.run(dag)
        layout = pass_.property_set['layout']

        for i in range(3):
            self.assertEqual(layout[qr[i]], i)

    def test_9q_circuit_16q_coupling(self):
        """Test finds trivial layout for 9q circuit with 2 registers on 16q device.
        """
        qr0 = QuantumRegister(4, 'q0')
        qr1 = QuantumRegister(5, 'q1')
        cr = ClassicalRegister(2, 'c')
        circuit = QuantumCircuit(qr0, qr1, cr)
        circuit.cx(qr0[1], qr0[2])
        circuit.cx(qr0[0], qr1[3])
        circuit.cx(qr1[4], qr0[2])
        circuit.measure(qr1[1], cr[0])
        circuit.measure(qr0[2], cr[1])

        dag = circuit_to_dag(circuit)
        pass_ = TrivialLayout(self.cmap16)
        pass_.run(dag)
        layout = pass_.property_set['layout']

        for i in range(4):
            self.assertEqual(layout[qr0[i]], i)
        for i in range(5):
            self.assertEqual(layout[qr1[i]], i+4)

    def test_raises_wider_circuit(self):
        """Test error is raised if the circuit is wider than coupling map.
        """
        qr0 = QuantumRegister(3, 'q0')
        qr1 = QuantumRegister(3, 'q1')
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr0, qr1)

        dag = circuit_to_dag(circuit)
        with self.assertRaises(TranspilerError):
            pass_ = TrivialLayout(self.cmap5)
            pass_.run(dag)


if __name__ == '__main__':
    unittest.main()
