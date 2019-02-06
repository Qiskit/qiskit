# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the ExtendLayout pass"""

import unittest

from qiskit import QuantumRegister
from qiskit.mapper import CouplingMap
from qiskit.transpiler.passes import ExtendLayout
from qiskit.mapper import Layout
from qiskit.test import QiskitTestCase


class TestExtendLayout(QiskitTestCase):
    """Tests the ExtendLayout pass"""

    def setUp(self):
        self.cmap5 = CouplingMap([[1, 0], [2, 0], [2, 1], [3, 2], [3, 4], [4, 2]])

    def test_3q_circuit_5q_coupling(self):
        """ Extends a 3q into a 5q

                    0 -> q0
        q0 -> 0     1 -> q1
        q1 -> 1  => 2 -> q2
        q2 -> 2     3 -> None
                    4 -> None
        """
        qr = QuantumRegister(3, 'q')
        initial_layout = Layout()
        initial_layout[0] = (qr, 0)
        initial_layout[1] = (qr, 1)
        initial_layout[2] = (qr, 2)

        pass_ = ExtendLayout(self.cmap5)
        pass_.property_set['layout'] = initial_layout

        pass_.run(None)
        after_layout = pass_.property_set['layout']

        self.assertEqual(after_layout[0], qr[0])
        self.assertEqual(after_layout[1], qr[1])
        self.assertEqual(after_layout[2], qr[2])
        self.assertEqual(after_layout[3], None)
        self.assertEqual(after_layout[4], None)

    def test_3q_with_holes_5q_coupling(self):
        """ Extends a 3q with holes into a 5q

                       0 -> q0
        q0 -> 0        1 -> None
        None -> 1   => 2 ->  q2
        q2 -> 2        3 -> None
                       4 -> None
        """
        qr = QuantumRegister(3, 'q')
        initial_layout = Layout()
        initial_layout[0] = (qr, 0)
        initial_layout[1] = None
        initial_layout[2] = (qr, 2)

        pass_ = ExtendLayout(self.cmap5)
        pass_.property_set['layout'] = initial_layout
        pass_.run(None)
        after_layout = pass_.property_set['layout']

        self.assertEqual(after_layout[0], qr[0])
        self.assertEqual(after_layout[1], None)
        self.assertEqual(after_layout[2], qr[2])
        self.assertEqual(after_layout[3], None)
        self.assertEqual(after_layout[4], None)

    def test_3q_out_of_order_5q_coupling(self):
        """ Extends a 3q which is out of order into a 5q

                       0 <- q0
        q0 -> 0        1 <- None
        q1 -> 3   =>   2 <- q2
        q2 -> 2        3 <- q1
                       4 <- None
        """
        qr = QuantumRegister(3, 'q')
        initial_layout = Layout()
        initial_layout[0] = (qr, 0)
        initial_layout[3] = (qr, 1)
        initial_layout[2] = (qr, 2)

        pass_ = ExtendLayout(self.cmap5)
        pass_.property_set['layout'] = initial_layout
        pass_.run(None)
        after_layout = pass_.property_set['layout']

        self.assertEqual(after_layout[0], qr[0])
        self.assertEqual(after_layout[1], None)
        self.assertEqual(after_layout[2], qr[2])
        self.assertEqual(after_layout[3], qr[1])
        self.assertEqual(after_layout[4], None)


if __name__ == '__main__':
    unittest.main()
