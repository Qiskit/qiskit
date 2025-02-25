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
import copy

from qiskit.circuit import bit, register
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestBitClass(QiskitTestCase):
    """Test library of boolean logic quantum circuits."""

    def test_bit_eq_invalid_type_comparison(self):
        orig_reg = register.Register(3)
        test_bit = bit.Bit(orig_reg, 0)
        self.assertNotEqual(test_bit, 3.14)

    def test_old_style_bit_equality(self):
        test_reg = register.Register(size=3, name="foo")

        self.assertEqual(bit.Bit(test_reg, 0), bit.Bit(test_reg, 0))
        self.assertNotEqual(bit.Bit(test_reg, 0), bit.Bit(test_reg, 2))

        reg_copy = copy.copy(test_reg)

        self.assertEqual(bit.Bit(test_reg, 0), bit.Bit(reg_copy, 0))
        self.assertNotEqual(bit.Bit(test_reg, 0), bit.Bit(reg_copy, 1))

        reg_larger = register.Register(size=4, name="foo")

        self.assertNotEqual(bit.Bit(test_reg, 0), bit.Bit(reg_larger, 0))

        reg_renamed = register.Register(size=3, name="bar")

        self.assertNotEqual(bit.Bit(test_reg, 0), bit.Bit(reg_renamed, 0))

        reg_difftype = register.Register(size=3, name="bar")

        self.assertNotEqual(bit.Bit(test_reg, 0), bit.Bit(reg_difftype, 0))

    def test_old_style_bit_deepcopy(self):
        """Verify deep-copies of bits are equal but not the same instance."""
        test_reg = register.Register(size=3, name="foo")

        bit1 = bit.Bit(test_reg, 0)
        bit2 = copy.deepcopy(bit1)

        self.assertIsNot(bit1, bit2)
        self.assertIsNot(bit1._register, bit2._register)
        self.assertEqual(bit1, bit2)

    def test_old_style_bit_copy(self):
        """Verify copies of bits are the same instance."""
        bit1 = bit.Bit()
        bit2 = copy.copy(bit1)

        self.assertIs(bit1, bit2)


class TestNewStyleBit(QiskitTestCase):
    """Test behavior of new-style bits."""

    def test_bits_do_not_require_registers(self):
        """Verify we can create a bit outside the context of a register."""
        self.assertIsInstance(bit.Bit(), bit.Bit)

    def test_new_style_bit_deepcopy(self):
        """Verify deep-copies of bits are the same instance."""
        bit1 = bit.Bit()
        bit2 = copy.deepcopy(bit1)

        self.assertIs(bit1, bit2)

    def test_new_style_bit_copy(self):
        """Verify copies of bits are the same instance."""
        bit1 = bit.Bit()
        bit2 = copy.copy(bit1)

        self.assertIs(bit1, bit2)

    def test_new_style_bit_equality(self):
        """Verify bits instances are equal only to themselves."""
        bit1 = bit.Bit()
        bit2 = bit.Bit()

        self.assertEqual(bit1, bit1)
        self.assertNotEqual(bit1, bit2)
        self.assertNotEqual(bit1, 3.14)
