# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" Tests the layout object"""

import unittest

from qiskit.mapper import Layout
from qiskit.mapper._layout import LayoutError
from .common import QiskitTestCase


class LayoutTest(QiskitTestCase):
    """Test the methods in the layout object."""

    def test_layout_from_dict(self):
        """Constructor from a dict"""
        layout = Layout({('qr', 0): 0,
                         ('qr', 1): 1,
                         ('qr', 2): 2})

        self.assertEqual(layout[('qr', 0)], 0)
        self.assertEqual(layout[('qr', 1)], 1)
        self.assertEqual(layout[('qr', 2)], 2)
        self.assertEqual(layout[0], ('qr', 0))
        self.assertEqual(layout[1], ('qr', 1))
        self.assertEqual(layout[2], ('qr', 2))

    def test_layout_from_list(self):
        """Constructor from a list"""
        layout = Layout([('q0', 0), ('q1', 0), None, ('q1', 1), ('q0', 1)])

        self.assertEqual(layout[('q0', 0)], 0)
        self.assertEqual(layout[('q1', 0)], 1)
        with self.assertRaises(KeyError):
            layout[None]
        self.assertEqual(layout[('q1', 1)], 3)
        self.assertEqual(layout[('q0', 1)], 4)

        self.assertEqual(layout[0], ('q0', 0))
        self.assertEqual(layout[1], ('q1', 0))
        self.assertEqual(layout[2], None)
        self.assertEqual(layout[3], ('q1', 1))
        self.assertEqual(layout[4], ('q0', 1))

    def test_layout_set(self):
        """Setter"""
        layout = Layout()
        layout[('qr', 0)] = 0
        self.assertEqual(layout[('qr', 0)], 0)
        self.assertEqual(layout[0], ('qr', 0))

    def test_layout_len(self):
        """Length of the layout is the amount of wires"""
        layout = Layout()
        self.assertEqual(len(layout), 0)
        layout.add(('qr', 0))
        self.assertEqual(len(layout), 1)
        layout.add(('qr', 1), 3)
        self.assertEqual(len(layout), 4)

    def test_layout_set_len(self):
        """Length setter"""
        layout = Layout()
        layout.add(('qr', 1), 3)
        layout.length(4)
        self.assertEqual(len(layout), 4)

    def test_layout_idle_wires(self):
        """Get wires that are not mapped"""
        layout = Layout()
        layout.add(('qr', 1), 2)
        layout.length(4)
        self.assertEqual(layout.idle_wires(), [0, 1, 3])

    def test_layout_get_bits(self):
        """Get the map from the logical (qu)bits view"""
        layout_dict = {('qr', 0): 0,
                       ('qr', 1): 1,
                       ('qr', 2): 2}
        layout = Layout(layout_dict)
        self.assertDictEqual(layout_dict, layout.get_bits())

    def test_layout_get_wires(self):
        """Get the map from the physical wires view"""
        layout = Layout({('qr', 0): 0, ('qr', 1): 1, ('qr', 2): 2})
        self.assertDictEqual(layout.get_wires(), {0: ('qr', 0), 1: ('qr', 1), 2: ('qr', 2)})

    def test_layout_add(self):
        """add() method"""
        layout = Layout()
        layout[('qr,0')] = 0
        layout.add(('qr', 1))

        self.assertEqual(layout[('qr', 1)], 1)
        self.assertEqual(layout[1], ('qr', 1))

    def test_layout_swap(self):
        """swap() method"""
        layout = Layout()
        layout.add(('qr', 0))
        layout.add(('qr', 1))
        layout.swap(0, 1)
        self.assertDictEqual(layout.get_bits(), {('qr', 0): 1, ('qr', 1): 0})

    def test_layout_swap_error(self):
        """swap() method"""
        layout = Layout()
        layout.add(('qr', 0))
        layout.add(('qr', 1))
        with self.assertRaises(LayoutError):
            layout.swap(0, ('qr', 0))

if __name__ == '__main__':
    unittest.main()
