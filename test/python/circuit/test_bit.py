# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-function-docstring

"""Test library of quantum circuits."""

from unittest import mock

from qiskit.test import QiskitTestCase
from qiskit.circuit import bit, QuantumRegister


class TestNewStyleBit(QiskitTestCase):
    """Test behavior of new-style bits."""

    def test_bits_do_not_require_registers(self):
        """Verify we can create a bit outside the context of a register."""
        self.assertIsInstance(bit.Bit(), bit.Bit)

    def test_newstyle_bit_equality(self):
        """Verify bits instances are equal only to themselves."""
        bit1 = bit.Bit()
        bit2 = bit.Bit()

        self.assertEqual(bit1, bit1)
        self.assertNotEqual(bit1, bit2)
        self.assertNotEqual(bit1, 3.14)

    def test_bit_register_backreferences_deprecated(self):
        """Verify we raise a deprecation warning for register back-references."""

        qr = QuantumRegister(3, "test_qr")
        qubit = qr[0]

        with self.assertWarnsRegex(DeprecationWarning, "deprecated"):
            _ = qubit.index

        with self.assertWarnsRegex(DeprecationWarning, "deprecated"):
            _ = qubit.register
