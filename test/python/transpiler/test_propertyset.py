# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Transpiler PropertySet testing"""

from ..common import QiskitTestCase
from qiskit.transpiler import PropertySet


class TestPropertySet(QiskitTestCase):

    def setUp(self):
        self.ps = PropertySet()

    def test_get_non_existent(self):
        self.assertIsNone(self.ps['does_not_exists'])

    def test_get_set_and_retrive(self):
        self.ps['property'] = 'value'
        self.assertEqual(self.ps['property'], 'value')

    def test_fixed_point_non_existent(self):
        self.assertFalse(self.ps.fixed_point('does_not_exist'))

    def test_fixed_point_setting_to_none(self):
        self.ps['property'] = None
        self.ps['property'] = None
        self.assertFalse(self.ps.fixed_point('property'))

    def test_fixed_point_reached(self):
        self.ps['property'] = 1
        self.assertFalse(self.ps.fixed_point('property'))
        self.ps['property'] = 1
        self.assertTrue(self.ps.fixed_point('property'))

    def test_fixed_point_not_reached(self):
        self.ps['property'] = 1
        self.assertFalse(self.ps.fixed_point('property'))
        self.ps['property'] = 2
        self.assertFalse(self.ps.fixed_point('property'))

    def test_fixed_point_left(self):
        self.ps['property'] = 1
        self.assertFalse(self.ps.fixed_point('property'))
        self.ps['property'] = 1
        self.assertTrue(self.ps.fixed_point('property'))
        self.ps['property'] = 2
        self.assertFalse(self.ps.fixed_point('property'))


if __name__ == '__main__':
    unittest.main()
