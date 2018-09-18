# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Transpiler PropertySet testing"""

import unittest
import logging
from qiskit.transpiler import PropertySet
from qiskit.transpiler._propertysetutilities import fixed_point
from ..common import QiskitTestCase

logger = "LocalLogger"


def dummy_utility(property_set, key, new_value):
    """ A dummy utility just for testing the utility registration"""
    logging.getLogger(logger).info('the property %s is updated with %s (previously %s)',
                                       key, new_value, property_set[key])


class TestPropertySet(QiskitTestCase):
    """ Tests for PropertySet methods. """

    def setUp(self):
        self.pset = PropertySet()
        self.pset.add_utility(fixed_point)

    def test_get_non_existent(self):
        """ Getting non-existent property should return None. """
        self.assertIsNone(self.pset['does_not_exists'])

    def test_get_set_and_retrive(self):
        """ Setting and retrieving."""
        self.pset['property'] = 'value'
        self.assertEqual(self.pset['property'], 'value')

    def test_fixed_point_setting_to_none(self):
        """ Setting a property to None twice does not create a fixed-point. """
        self.pset['property'] = None
        self.pset['property'] = None
        self.assertFalse(self.pset['fixed_point']['property'])

    def test_fixed_point_reached(self):
        """ Setting a property to the same value twice creates a fixed-point. """
        self.pset['property'] = 1
        self.assertFalse(self.pset['fixed_point']['property'])
        self.pset['property'] = 1
        self.assertTrue(self.pset['fixed_point']['property'])

    def test_fixed_point_not_reached(self):
        """ Setting a property with different values does not create a fixed-point. """
        self.pset['property'] = 1
        self.assertFalse(self.pset['fixed_point']['property'])
        self.pset['property'] = 2
        self.assertFalse(self.pset['fixed_point']['property'])

    def test_fixed_point_left(self):
        """ A fixed-point is not permanent. """
        self.pset['property'] = 1
        self.assertFalse(self.pset['fixed_point']['property'])
        self.pset['property'] = 1
        self.assertTrue(self.pset['fixed_point']['property'])
        self.pset['property'] = 2
        self.assertFalse(self.pset['fixed_point']['property'])

    def test_dummy_utility(self):
        """ Testing add_utility, on_change and getter in a dummy utility """
        self.pset.add_utility(dummy_utility)
        with self.assertLogs(logger, level='INFO') as context:
            self.pset['property'] = 1
            self.pset['property'] = 2
        self.assertEqual([record.message for record in context.records],
                         ['the property property is updated with 1 (previously None)',
                          'the property property is updated with 2 (previously 1)'])

if __name__ == '__main__':
    unittest.main()
