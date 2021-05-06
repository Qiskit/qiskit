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


class TestBitClass(QiskitTestCase):
    """Test library of boolean logic quantum circuits."""

    def test_bit_eq_invalid_type_comparison(self):
        orig_reg = mock.MagicMock()
        orig_reg.size = 3
        test_bit = bit.Bit(orig_reg, 0)
        self.assertNotEqual(test_bit, 3.14)

    def test_old_style_bit_equality(self):
        test_reg = mock.MagicMock(size=3, name="foo")
        test_reg.__str__.return_value = "Register(3, 'foo')"

        self.assertEqual(bit.Bit(test_reg, 0), bit.Bit(test_reg, 0))
        self.assertNotEqual(bit.Bit(test_reg, 0), bit.Bit(test_reg, 2))

        reg_copy = mock.MagicMock(size=3, name="foo")
        reg_copy.__str__.return_value = "Register(3, 'foo')"

        self.assertEqual(bit.Bit(test_reg, 0), bit.Bit(reg_copy, 0))
        self.assertNotEqual(bit.Bit(test_reg, 0), bit.Bit(reg_copy, 1))

        reg_larger = mock.MagicMock(size=4, name="foo")
        reg_larger.__str__.return_value = "Register(4, 'foo')"

        self.assertNotEqual(bit.Bit(test_reg, 0), bit.Bit(reg_larger, 0))

        reg_renamed = mock.MagicMock(size=3, name="bar")
        reg_renamed.__str__.return_value = "Register(3, 'bar')"

        self.assertNotEqual(bit.Bit(test_reg, 0), bit.Bit(reg_renamed, 0))

        reg_difftype = mock.MagicMock(size=3, name="bar")
        reg_difftype.__str__.return_value = "QuantumRegister(3, 'bar')"

        self.assertNotEqual(bit.Bit(test_reg, 0), bit.Bit(reg_difftype, 0))


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
