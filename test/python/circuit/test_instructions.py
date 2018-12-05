# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=unused-import

"""Test Qiskit's Instruction class."""

import os
import tempfile
import unittest

from qiskit import QuantumRegister, ClassicalRegister
from qiskit._instruction import Instruction
from qiskit.extensions.standard.h import HGate
from qiskit.extensions.standard.cx import CnotGate
from ..common import QiskitTestCase


class TestInstructions(QiskitTestCase):
    """Instructions tests."""

    def test_instructions_equal(self):
        """Test equality of two instructions.
        """
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        hop1 = Instruction('h', [], qr, cr)
        hop2 = Instruction('s', [], qr, cr)
        hop3 = Instruction('h', [], qr, cr)

        uop1 = Instruction('u', [0.4, 0.5, 0.5], qr, cr)
        uop2 = Instruction('u', [0.4, 0.6, 0.5], qr, cr)
        uop3 = Instruction('v', [0.4, 0.5, 0.5], qr, cr)
        uop4 = Instruction('u', [0.4, 0.5, 0.5], qr, cr)
        self.assertFalse(hop1 == hop2)
        self.assertTrue(hop1 == hop3)
        self.assertFalse(uop1 == uop2)
        self.assertTrue(uop1 == uop4)
        self.assertFalse(uop1 == uop3)
        self.assertTrue(HGate(qr[0]) == HGate(qr[1]))
        self.assertFalse(HGate(qr[0]) == CnotGate(qr[0], qr[1]))
        self.assertFalse(hop1 == HGate(qr[2]))


if __name__ == '__main__':
    unittest.main()
