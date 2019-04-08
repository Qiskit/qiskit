# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""FixedPoint pass testing"""

import unittest
from qiskit.transpiler.passes import FixedPoint
from qiskit.test import QiskitTestCase


class TestFixedPointPass(QiskitTestCase):
    """ Tests for FixedPoint pass. """

    def setUp(self):
        self.pass_ = FixedPoint('property')
        self.pset = self.pass_.property_set
        self.dag = None  # The pass do not read the DAG.

    def test_fixed_point_setting_to_none(self):
        """ Setting a property to None twice does not create a fixed-point. """
        self.pass_.property_set['property'] = None
        self.pass_.run(self.dag)
        self.pass_.property_set['property'] = None
        self.pass_.run(self.dag)
        self.assertFalse(self.pset['property_fixed_point'])

    def test_fixed_point_reached(self):
        """ Setting a property to the same value twice creates a fixed-point. """
        self.pset['property'] = 1
        self.pass_.run(self.dag)
        self.assertFalse(self.pset['property_fixed_point'])
        self.pset['property'] = 1
        self.pass_.run(self.dag)
        self.assertTrue(self.pset['property_fixed_point'])

    def test_fixed_point_not_reached(self):
        """ Setting a property with different values does not create a fixed-point. """
        self.pset['property'] = 1
        self.pass_.run(self.dag)
        self.assertFalse(self.pset['property_fixed_point'])
        self.pset['property'] = 2
        self.pass_.run(self.dag)
        self.assertFalse(self.pset['property_fixed_point'])

    def test_fixed_point_left(self):
        """ A fixed-point is not permanent. """
        self.pset['property'] = 1
        self.pass_.run(self.dag)
        self.assertFalse(self.pset['property_fixed_point'])
        self.pset['property'] = 1
        self.pass_.run(self.dag)
        self.assertTrue(self.pset['property_fixed_point'])
        self.pset['property'] = 2
        self.pass_.run(self.dag)
        self.assertFalse(self.pset['property_fixed_point'])


if __name__ == '__main__':
    unittest.main()
