# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" Tests the layout object"""

import unittest

from qiskit import QuantumRegister
from qiskit.mapper import Layout
from qiskit.mapper._layout import LayoutError
from .common import QiskitTestCase


class LayoutTest(QiskitTestCase):
    """Test the methods in the layout object."""

    def setUp(self):
        self.qr = QuantumRegister(3, 'qr')

    def test_layout_from_dict(self):
        """Constructor from a dict"""
        layout = Layout({(self.qr, 0): 0,
                         (self.qr, 1): 1,
                         (self.qr, 2): 2})

        self.assertEqual(layout[(self.qr, 0)], 0)
        self.assertEqual(layout[(self.qr, 1)], 1)
        self.assertEqual(layout[(self.qr, 2)], 2)
        self.assertEqual(layout[0], (self.qr, 0))
        self.assertEqual(layout[1], (self.qr, 1))
        self.assertEqual(layout[2], (self.qr, 2))

    def test_layout_from_list(self):
        """Constructor from a list"""
        layout = Layout([(QuantumRegister(2, 'q0'), 0),
                         (QuantumRegister(2, 'q1'), 0),
                         None,
                         (QuantumRegister(2, 'q1'), 1),
                         (QuantumRegister(2, 'q0'), 1)])

        self.assertEqual(layout[(QuantumRegister(2, 'q0'), 0)], 0)
        self.assertEqual(layout[(QuantumRegister(2, 'q1'), 0)], 1)
        with self.assertRaises(KeyError):
            _ = layout[None]
        self.assertEqual(layout[(QuantumRegister(2, 'q1'), 1)], 3)
        self.assertEqual(layout[(QuantumRegister(2, 'q0'), 1)], 4)

        self.assertEqual(layout[0], (QuantumRegister(2, 'q0'), 0))
        self.assertEqual(layout[1], (QuantumRegister(2, 'q1'), 0))
        self.assertEqual(layout[2], None)
        self.assertEqual(layout[3], (QuantumRegister(2, 'q1'), 1))
        self.assertEqual(layout[4], (QuantumRegister(2, 'q0'), 1))

    def test_layout_set(self):
        """Setter"""
        layout = Layout()
        layout[(self.qr, 0)] = 0
        self.assertEqual(layout[(self.qr, 0)], 0)
        self.assertEqual(layout[0], (self.qr, 0))

    def test_layout_len(self):
        """Length of the layout is the amount of wires"""
        layout = Layout()
        self.assertEqual(len(layout), 0)
        layout.add((self.qr, 0))
        self.assertEqual(len(layout), 1)
        layout.add((self.qr, 1), 3)
        self.assertEqual(len(layout), 4)

    def test_layout_set_len(self):
        """Length setter"""
        layout = Layout()
        layout.add((self.qr, 1), 3)
        layout.set_length(4)
        self.assertEqual(len(layout), 4)

    def test_layout_idle_wires(self):
        """Get wires that are not mapped"""
        layout = Layout()
        layout.add((self.qr, 1), 2)
        layout.set_length(4)
        self.assertEqual(layout.idle_wires(), [0, 1, 3])

    def test_layout_get_bits(self):
        """Get the map from the (qu)bits view"""
        layout_dict = {(self.qr, 0): 0,
                       (self.qr, 1): 1,
                       (self.qr, 2): 2}
        layout = Layout(layout_dict)
        self.assertDictEqual(layout_dict, layout.get_bits())

    def test_layout_get_wires(self):
        """Get the map from the wires view"""
        layout = Layout({(self.qr, 0): 0, (self.qr, 1): 1, (self.qr, 2): 2})
        self.assertDictEqual(layout.get_wires(), {0: (self.qr, 0),
                                                  1: (self.qr, 1),
                                                  2: (self.qr, 2)})

    def test_layout_add(self):
        """add() method"""
        layout = Layout()
        layout[(self.qr, 0)] = 0
        layout.add((self.qr, 1))

        self.assertEqual(layout[(self.qr, 1)], 1)
        self.assertEqual(layout[1], (self.qr, 1))

    def test_layout_add_register(self):
        """add_register() method"""
        layout = Layout()
        layout.add_register(QuantumRegister(2, 'q0'))
        layout.add_register(QuantumRegister(1, 'q1'))

        self.assertEqual(layout[(QuantumRegister(2, 'q0'), 0)], 0)
        self.assertEqual(layout[(QuantumRegister(2, 'q0'), 1)], 1)
        self.assertEqual(layout[(QuantumRegister(1, 'q1'), 0)], 2)

    def test_layout_swap(self):
        """swap() method"""
        layout = Layout()
        layout.add((self.qr, 0))
        layout.add((self.qr, 1))
        layout.swap(0, 1)
        self.assertDictEqual(layout.get_bits(), {(self.qr, 0): 1, (self.qr, 1): 0})

    def test_layout_swap_error(self):
        """swap() method error"""
        layout = Layout()
        layout.add((self.qr, 0))
        layout.add((self.qr, 1))
        with self.assertRaises(LayoutError):
            layout.swap(0, (self.qr, 0))


if __name__ == '__main__':
    unittest.main()
