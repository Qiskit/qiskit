# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""BasePass and generic pass testing"""

import unittest.mock
from ._dummy_passes import DummyAP, DummyTP, PassD_TP_NR_NP
from ..common import QiskitTestCase
from qiskit.transpiler import TranspilerUnknownOption

class TestGenericPass(QiskitTestCase):
    """ Passes have common caracteristics defined in BasePass."""

    def test_pass_setting(self):
        """ Passes can be set via `set`."""
        tp_pass = DummyTP()

        self.assertTrue(tp_pass.idempotence)        # By default, passes are idempotent
        self.assertFalse(tp_pass.ignore_preserves)  # By default, passes do not ignore preserves
        self.assertFalse(tp_pass.ignore_requires)   # By default, passes do not ignore requires

        tp_pass.set(idempotence=False, ignore_preserves=True, ignore_requires=True)

        self.assertFalse(tp_pass.idempotence)
        self.assertTrue(tp_pass.ignore_requires)
        self.assertTrue(tp_pass.ignore_preserves)

    def test_pass_unknown_option(self):
        """ An option in a pass should be in the set of allowed options. """
        tp_pass = DummyTP()
        with self.assertRaises(TranspilerUnknownOption):
            tp_pass.set(not_an_option=False)

    def test_is_TP_or_AP(self):
        """ Passes have isTransformationPass and isAnalysisPass properties."""
        tp_pass = DummyTP()
        self.assertTrue(tp_pass.isTransformationPass)
        self.assertFalse(tp_pass.isAnalysisPass)
        ap_pass = DummyAP()
        self.assertFalse(ap_pass.isTransformationPass)
        self.assertTrue(ap_pass.isAnalysisPass)

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
