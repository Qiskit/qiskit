# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test Qiskit's Instruction class."""

import unittest

from qiskit.circuit import Instruction
from qiskit.extensions.standard.h import HGate
from qiskit.extensions.standard.cx import CnotGate
from qiskit.test import QiskitTestCase


class TestInstructions(QiskitTestCase):
    """Instructions tests."""

    def test_instructions_equal(self):
        """Test equality of two instructions.
        """
        hop1 = Instruction('h', [])
        hop2 = Instruction('s', [])
        hop3 = Instruction('h', [])

        uop1 = Instruction('u', [0.4, 0.5, 0.5])
        uop2 = Instruction('u', [0.4, 0.6, 0.5])
        uop3 = Instruction('v', [0.4, 0.5, 0.5])
        uop4 = Instruction('u', [0.4, 0.5, 0.5])
        self.assertFalse(hop1 == hop2)
        self.assertTrue(hop1 == hop3)
        self.assertFalse(uop1 == uop2)
        self.assertTrue(uop1 == uop4)
        self.assertFalse(uop1 == uop3)
        self.assertTrue(HGate() == HGate())
        self.assertFalse(HGate() == CnotGate())
        self.assertFalse(hop1 == HGate())


if __name__ == '__main__':
    unittest.main()
