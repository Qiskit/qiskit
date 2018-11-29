# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""FixedPoint pass testing"""

import unittest
from qiskit.transpiler import PropertySet
from qiskit.transpiler.passes import FixedPoint
from ..common import QiskitTestCase


class TestFixedPointPass(QiskitTestCase):
    """ Tests for PropertySet methods. """

    def setUp(self):
        self.pass_ = FixedPoint('property')
        self.pset = self.pass_.property_set = PropertySet()
        self.dag = None  # The pass do not read the DAG.

    def test_requires_field_none(self):
        """ When pass_that_updates_the_property is not passed, there are no requirements. """
        self.assertEqual(len(self.pass_.requires), 0)

    def test_fixed_point_property_is_created(self):
        """ The property set does not have a property called "fixed_point" and it is created after
        the  first run of the pass. """
        self.assertIsNone(self.pset['fixed_point'])
        self.pass_.run(self.dag)
        self.assertIsNotNone(self.pset['fixed_point'])

    def test_fixed_point_setting_to_none(self):
        """ Setting a property to None twice does not create a fixed-point. """
        self.pass_.property_set['property'] = None
        self.pass_.run(self.dag)
        self.pass_.property_set['property'] = None
        self.pass_.run(self.dag)
        self.assertFalse(self.pset['fixed_point']['property'])

    def test_fixed_point_reached(self):
        """ Setting a property to the same value twice creates a fixed-point. """
        self.pset['property'] = 1
        self.pass_.run(self.dag)
        self.assertFalse(self.pset['fixed_point']['property'])
        self.pset['property'] = 1
        self.pass_.run(self.dag)
        self.assertTrue(self.pset['fixed_point']['property'])

    def test_fixed_point_not_reached(self):
        """ Setting a property with different values does not create a fixed-point. """
        self.pset['property'] = 1
        self.pass_.run(self.dag)
        self.assertFalse(self.pset['fixed_point']['property'])
        self.pset['property'] = 2
        self.pass_.run(self.dag)
        self.assertFalse(self.pset['fixed_point']['property'])

    def test_fixed_point_left(self):
        """ A fixed-point is not permanent. """
        self.pset['property'] = 1
        self.pass_.run(self.dag)
        self.assertFalse(self.pset['fixed_point']['property'])
        self.pset['property'] = 1
        self.pass_.run(self.dag)
        self.assertTrue(self.pset['fixed_point']['property'])
        self.pset['property'] = 2
        self.pass_.run(self.dag)
        self.assertFalse(self.pset['fixed_point']['property'])


if __name__ == '__main__':
    unittest.main()
