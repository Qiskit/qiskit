# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""BasePass and generic pass testing"""

import unittest.mock
from ._dummy_passes import DummyAP, DummyTP
from ..common import QiskitTestCase

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

    def test_is_TP_or_AP(self):
        """ Passes have isTransformationPass and isAnalysisPass properties."""
        tp_pass = DummyTP()
        self.assertTrue(tp_pass.isTransformationPass)
        self.assertFalse(tp_pass.isAnalysisPass)
        ap_pass = DummyAP()
        self.assertFalse(ap_pass.isTransformationPass)
        self.assertTrue(ap_pass.isAnalysisPass)

if __name__ == '__main__':
    unittest.main()
