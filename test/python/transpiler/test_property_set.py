# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Transpiler PropertySet testing"""

import unittest
from qiskit.transpiler import PropertySet
from qiskit.test import QiskitTestCase


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

    def test_str(self):
        """ Test __str__ method. """
        self.pset['property'] = 'value'
        self.assertEqual(str(self.pset), "{'property': 'value'}")

    def test_repr(self):
        """ Test __repr__ method. """
        self.pset['property'] = 'value'
        self.assertEqual(str(repr(self.pset)), "{'property': 'value'}")


if __name__ == '__main__':
    unittest.main()
