# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Transpiler PropertySet testing"""

import unittest
from qiskit.transpiler import PropertySet
from ..common import QiskitTestCase


class TestPropertySet(QiskitTestCase):
    """ Tests for PropertySet methods. """
    def setUp(self):
        self.pset = PropertySet()

    def test_get_non_existent(self):
        """ Getting non-existent property should return None. """
        self.assertIsNone(self.pset['does_not_exists'])

    def test_get_set_and_retrive(self):
        """ Setting and retrieving."""
        self.pset['property'] = 'value'
        self.assertEqual(self.pset['property'], 'value')

    def test_fixed_point_non_existent(self):
        """ Getting the fixed point of a non-existent property should return False. """
        self.assertFalse(self.pset.fixed_point('does_not_exist'))

    def test_fixed_point_setting_to_none(self):
        """ Setting a property to None twice does not create a fixed-point. """
        self.pset['property'] = None
        self.pset['property'] = None
        self.assertFalse(self.pset.fixed_point('property'))

    def test_fixed_point_reached(self):
        """ Setting a property to the same value twice creates a fixed-point. """
        self.pset['property'] = 1
        self.assertFalse(self.pset.fixed_point('property'))
        self.pset['property'] = 1
        self.assertTrue(self.pset.fixed_point('property'))

    def test_fixed_point_not_reached(self):
        """ Setting a property with different values does not create a fixed-point. """
        self.pset['property'] = 1
        self.assertFalse(self.pset.fixed_point('property'))
        self.pset['property'] = 2
        self.assertFalse(self.pset.fixed_point('property'))

    def test_fixed_point_left(self):
        """ A fixed-point is not permanent. """
        self.pset['property'] = 1
        self.assertFalse(self.pset.fixed_point('property'))
        self.pset['property'] = 1
        self.assertTrue(self.pset.fixed_point('property'))
        self.pset['property'] = 2
        self.assertFalse(self.pset.fixed_point('property'))


if __name__ == '__main__':
    unittest.main()
