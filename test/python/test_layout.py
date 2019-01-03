# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" Tests the layout object"""

import copy
import unittest

from qiskit import QuantumRegister
from qiskit.mapper import Layout
from qiskit.mapper.exceptions import LayoutError
from qiskit.test import QiskitTestCase


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
        """Length of the layout is the amount of physical bits"""
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

    def test_layout_idle_physical_bits(self):
        """Get physical_bits that are not mapped"""
        layout = Layout()
        layout.add((self.qr, 1), 2)
        layout.set_length(4)
        self.assertEqual(layout.idle_physical_bits(), [0, 1, 3])

    def test_layout_get_bits(self):
        """Get the map from the (qu)bits view"""
        layout_dict = {(self.qr, 0): 0,
                       (self.qr, 1): 1,
                       (self.qr, 2): 2}
        layout = Layout(layout_dict)
        self.assertDictEqual(layout_dict, layout.get_virtual_bits())

    def test_layout_get_physical_bits(self):
        """Get the map from the physical bits view"""
        layout = Layout({(self.qr, 0): 0, (self.qr, 1): 1, (self.qr, 2): 2})
        self.assertDictEqual(layout.get_physical_bits(), {0: (self.qr, 0),
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
        self.assertDictEqual(layout.get_virtual_bits(), {(self.qr, 0): 1, (self.qr, 1): 0})

    def test_layout_swap_error(self):
        """swap() method error"""
        layout = Layout()
        layout.add((self.qr, 0))
        layout.add((self.qr, 1))
        with self.assertRaises(LayoutError):
            layout.swap(0, (self.qr, 0))

    def test_layout_combine(self):
        """combine_into_edge_map() method"""
        layout = Layout()
        layout.add((self.qr, 0))
        layout.add((self.qr, 1))
        another_layout = Layout()
        another_layout.add((self.qr, 1))
        another_layout.add((self.qr, 0))

        edge_map = layout.combine_into_edge_map(another_layout)
        self.assertDictEqual(edge_map, {(self.qr, 0): (self.qr, 1), (self.qr, 1): (self.qr, 0)})

    def test_layout_combine_bigger(self):
        """combine_into_edge_map() method with another_layout is bigger"""
        layout = Layout()
        layout.add((self.qr, 0))
        layout.add((self.qr, 1))
        another_layout = Layout()
        another_layout.add((self.qr, 1))
        another_layout.add((self.qr, 0))
        another_layout.add((self.qr, 2))

        edge_map = layout.combine_into_edge_map(another_layout)
        self.assertDictEqual(edge_map, {(self.qr, 0): (self.qr, 1), (self.qr, 1): (self.qr, 0)})

    def test_layout_combine_smaller(self):
        """combine_into_edge_map() method with another_layout is smaller and raises an Error"""
        layout = Layout()
        layout.add((self.qr, 0))
        layout.add((self.qr, 1))
        layout.add((self.qr, 2))
        another_layout = Layout()
        another_layout.add((self.qr, 1))
        another_layout.add((self.qr, 0))

        with self.assertRaises(LayoutError):
            _ = layout.combine_into_edge_map(another_layout)

    def test_copy(self):
        """Test copy methods return equivalent layouts."""
        layout = Layout()
        layout.add((self.qr, 0))
        layout.add((self.qr, 1))

        layout_dict_copy = layout.copy()
        self.assertTrue(isinstance(layout_dict_copy, Layout))
        self.assertDictEqual(layout, layout_dict_copy)

        layout_copy_copy = copy.copy(layout)
        self.assertTrue(isinstance(layout_copy_copy, Layout))
        self.assertDictEqual(layout, layout_copy_copy)

        layout_copy_deepcopy = copy.deepcopy(layout)
        self.assertTrue(isinstance(layout_copy_deepcopy, Layout))
        self.assertDictEqual(layout, layout_copy_deepcopy)


if __name__ == '__main__':
    unittest.main()
