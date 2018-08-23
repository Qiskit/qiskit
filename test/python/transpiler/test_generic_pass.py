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
        """ A single chain of passes, with Requests and Preserves."""
        tp_pass = DummyTP()
        self.assertTrue(tp_pass.idempotence)   # By default, passes are idempotent
        tp_pass.set(idempotence = False) # Set idempotence as False
        self.assertFalse(tp_pass.idempotence)

if __name__ == '__main__':
    unittest.main()
