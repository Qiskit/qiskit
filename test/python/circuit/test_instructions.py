# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=unused-import

"""Test Qiskit's QuantumCircuit class."""

import os
import tempfile
import unittest

from qiskit import QuantumRegister, ClassicalRegister
from qiskit._instruction import Instruction
from qiskit.extensions.standard import h, cx
from ..common import QiskitTestCase


class TestInstructions(QiskitTestCase):
    """Instructions tests."""

    def test_instructions_equal(self):
        """Test equality of two instructions.
        """
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        h1 = Instruction('h', [], qr, cr)
        h2 = Instruction('s', [], qr, cr)
        h3 = Instruction('h', [], qr, cr)

        u1 = Instruction('u', [0.4, 0.5, 0.5], qr, cr)
        u2 = Instruction('u', [0.4, 0.6, 0.5], qr, cr)
        u3 = Instruction('v', [0.4, 0.5, 0.5], qr, cr)
        u4 = Instruction('u', [0.4, 0.5, 0.5], qr, cr)
        self.assertFalse(h1 == h2)
        self.assertTrue(h1 == h3)
        self.assertFalse(u1 == u2)
        self.assertTrue(u1 == u4)
        self.assertFalse(u1 == u3)
        self.assertTrue(h == h)
        self.assertFalse(h == cx)
        self.assertFalse(h1 == h)


if __name__ == '__main__':
    unittest.main()
