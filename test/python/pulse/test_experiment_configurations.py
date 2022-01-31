# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test cases for the experimental conditions for pulse."""
import unittest

from qiskit.pulse.channels import DriveChannel, MeasureChannel, AcquireChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse import LoConfig, LoRange
from qiskit.test import QiskitTestCase


class TestLoRange(QiskitTestCase):
    """Test LO LoRange."""

    def test_properties_includes_and_eq(self):
        """Test creation of LoRange. Test upper/lower bounds and includes.
        Test __eq__ for two same and different LoRange's.
        """
        lo_range_1 = LoRange(lower_bound=-0.1, upper_bound=+0.1)

        self.assertEqual(lo_range_1.lower_bound, -0.1)
        self.assertEqual(lo_range_1.upper_bound, +0.1)
        self.assertTrue(lo_range_1.includes(0.0))

        lo_range_2 = LoRange(lower_bound=-0.1, upper_bound=+0.1)
        lo_range_3 = LoRange(lower_bound=-0.2, upper_bound=+0.2)

        self.assertTrue(lo_range_1 == lo_range_2)
        self.assertFalse(lo_range_1 == lo_range_3)


class TestLoConfig(QiskitTestCase):
    """LoConfig tests."""

    def test_can_create_empty_user_lo_config(self):
        """Test if a LoConfig can be created without no arguments."""
        user_lo_config = LoConfig()
        self.assertEqual({}, user_lo_config.qubit_los)
        self.assertEqual({}, user_lo_config.meas_los)

    def test_can_create_valid_user_lo_config(self):
        """Test if a LoConfig can be created with valid user_los."""
        channel1 = DriveChannel(0)
        channel2 = MeasureChannel(0)
        user_lo_config = LoConfig({channel1: 1.4, channel2: 3.6})
        self.assertEqual(1.4, user_lo_config.qubit_los[channel1])
        self.assertEqual(3.6, user_lo_config.meas_los[channel2])

    def test_fail_to_create_with_out_of_range_user_lo(self):
        """Test if a LoConfig cannot be created with invalid user_los."""
        channel = DriveChannel(0)
        with self.assertRaises(PulseError):
            LoConfig({channel: 3.3}, {channel: (1.0, 2.0)})

    def test_fail_to_create_with_invalid_channel(self):
        """Test if a LoConfig cannot be created with invalid channel."""
        channel = AcquireChannel(0)
        with self.assertRaises(PulseError):
            LoConfig({channel: 1.0})

    def test_keep_dict_unchanged_after_updating_the_dict_used_in_construction(self):
        """Test if a LoConfig keeps its dictionary unchanged even after
        the dictionary used in construction is updated.
        """
        channel = DriveChannel(0)
        original = {channel: 3.4}
        user_lo_config = LoConfig(original)
        self.assertEqual(3.4, user_lo_config.qubit_los[channel])
        original[channel] = 5.6
        self.assertEqual(3.4, user_lo_config.qubit_los[channel])

    def test_get_channel_lo(self):
        """Test retrieving channel lo from LO config."""
        channel = DriveChannel(0)
        lo_config = LoConfig({channel: 1.0})
        self.assertEqual(lo_config.channel_lo(channel), 1.0)

        channel = MeasureChannel(0)
        lo_config = LoConfig({channel: 2.0})
        self.assertEqual(lo_config.channel_lo(channel), 2.0)

        with self.assertRaises(PulseError):
            lo_config.channel_lo(MeasureChannel(1))


if __name__ == "__main__":
    unittest.main()
