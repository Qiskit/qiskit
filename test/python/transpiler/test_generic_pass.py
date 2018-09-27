# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""BasePass and generic pass testing"""

import unittest.mock
from qiskit.transpiler import TranspilerUnknownOption
from ._dummy_passes import DummyAP, DummyTP, DummyNI, PassD_TP_NR_NP
from ..common import QiskitTestCase


class TestGenericPass(QiskitTestCase):
    """ Passes have common characteristics defined in BasePass."""

    def test_pass_setting(self):
        """ Passes can be set via `set`."""
        tp_pass = DummyTP()

        self.assertFalse(tp_pass.ignore_preserves)  # By default, passes do not ignore preserves
        self.assertFalse(tp_pass.ignore_requires)  # By default, passes do not ignore requires
        self.assertEqual(1000, tp_pass.max_iteration)  # By default, max_iteration is set to 1000

        tp_pass.ignore_preserves = True
        tp_pass.ignore_requires = True
        tp_pass.max_iteration = 10

        self.assertTrue(tp_pass.ignore_requires)
        self.assertTrue(tp_pass.ignore_preserves)
        self.assertEqual(10, tp_pass.max_iteration)

    def test_is_TP_or_AP(self):
        """ Passes have is_transformation_pass and is_analysis_pass properties."""
        tp_pass = DummyTP()
        self.assertTrue(tp_pass.is_transformation_pass)
        self.assertFalse(tp_pass.is_analysis_pass)
        ap_pass = DummyAP()
        self.assertFalse(ap_pass.is_transformation_pass)
        self.assertTrue(ap_pass.is_analysis_pass)

    def test_pass_diff_args(self):
        """ Passes instances with different arguments are differnt """
        pass1 = PassD_TP_NR_NP(argument1=[1, 2])
        pass2 = PassD_TP_NR_NP(argument1=[2, 1])
        self.assertNotEqual(pass1, pass2)

    def test_pass_kwargs_out_of_order(self):
        """ Passes instances with same arguments (independently of the order) are the same"""
        pass1 = PassD_TP_NR_NP(argument1=1, argument2=2)
        pass2 = PassD_TP_NR_NP(argument2=2, argument1=1)
        self.assertEqual(pass1, pass2)


if __name__ == '__main__':
    unittest.main()
