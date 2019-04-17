# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test cases for the experimental conditions for pulse."""
import unittest

from qiskit.pulse.channels import DriveChannel, MeasureChannel, AcquireChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse import LoConfig
from qiskit.test import QiskitTestCase


class TestLoConfig(QiskitTestCase):
    """LoConfig tests."""

    def test_can_create_empty_user_lo_config(self):
        """Test if a LoConfig can be created without no arguments.
        """
        user_lo_config = LoConfig()
        self.assertEqual({}, user_lo_config._q_lo_freq)
        self.assertEqual({}, user_lo_config._m_lo_freq)

    def test_can_create_valid_user_lo_config(self):
        """Test if a LoConfig can be created with valid user_los.
        """
        channel1 = DriveChannel(0, lo_freq=1.2, lo_freq_range=(1.0, 2.0))
        channel2 = MeasureChannel(0, lo_freq=3.4, lo_freq_range=(3.0, 4.0))
        user_lo_config = LoConfig({channel1: 1.4, channel2: 3.6})
        self.assertEqual(1.4, user_lo_config._q_lo_freq[channel1])
        self.assertEqual(3.6, user_lo_config._m_lo_freq[channel2])

    def test_fail_to_create_with_out_of_range_user_lo(self):
        """Test if a LoConfig cannot be created with invalid user_los.
        """
        channel = DriveChannel(0, lo_freq=1.2, lo_freq_range=(1.0, 2.0))
        with self.assertRaises(PulseError):
            _ = LoConfig({channel: 3.3})

    def test_fail_to_create_with_invalid_channel(self):
        """Test if a LoConfig cannot be created with invalid channel.
        """
        channel = AcquireChannel(0)
        with self.assertRaises(PulseError):
            _ = LoConfig({channel: 1.0})

    def test_keep_dict_unchanged_after_updating_the_dict_used_in_construction(self):
        """Test if a LoConfig keeps its dictionary unchanged even after
        the dictionary used in construction is updated.
        """
        channel = DriveChannel(0, lo_freq=1.2)
        original = {channel: 3.4}
        user_lo_config = LoConfig(original)
        self.assertEqual(3.4, user_lo_config._q_lo_freq[channel])
        original[channel] = 5.6
        self.assertEqual(3.4, user_lo_config._q_lo_freq[channel])


if __name__ == '__main__':
    unittest.main()
