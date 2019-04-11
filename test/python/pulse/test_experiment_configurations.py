# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test cases for the experimental conditions for pulse."""
import unittest

from qiskit.pulse.channels import DriveChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import UserLoDict
from qiskit.test import QiskitTestCase


class TestUserLoDict(QiskitTestCase):
    """UserLoDict tests."""

    def test_can_create_empty_user_lo_dict(self):
        """Test if a UserLoDict can be created without no arguments.
        """
        user_lo_dict = UserLoDict()
        self.assertEqual({}, user_lo_dict._user_lo_dic)

    def test_can_create_valid_user_lo_dict(self):
        """Test if a UserLoDict can be created with valid user_los.
        """
        channel = DriveChannel(0, lo_frequency=1.2, lo_freq_range=(1.0, 2.0))
        user_lo_dict = UserLoDict({channel: 1.4})
        self.assertEqual(1.4, user_lo_dict._user_lo_dic[channel])

    def test_fail_to_create_with_out_of_range_user_lo(self):
        """Test if a UserLoDict cannot be created with invalid user_los.
        """
        channel = DriveChannel(0, lo_frequency=1.2, lo_freq_range=(1.0, 2.0))
        with self.assertRaises(PulseError):
            _ = UserLoDict({channel: 3.3})

    def test_keep_dict_unchanged_after_updating_the_dict_used_in_construction(self):
        """Test if a UserLoDict keeps its dictionary unchanged even after
        the dictionary used in construction is updated.
        """
        channel = DriveChannel(0, lo_frequency=1.2)
        original = {channel: 3.4}
        user_lo_dict = UserLoDict(original)
        self.assertEqual(3.4, user_lo_dict._user_lo_dic[channel])
        original[channel] = 5.6
        self.assertEqual(3.4, user_lo_dict._user_lo_dic[channel])


if __name__ == '__main__':
    unittest.main()
