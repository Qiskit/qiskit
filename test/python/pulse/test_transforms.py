# -*- coding: utf-8 -*-

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

"""Test cases for pulse transforms."""
from typing import Set

from qiskit.pulse import (Schedule, SamplePulse, DriveChannel, Play,
                          Gaussian, GaussianSquare, ConstantPulse, Drag)
from qiskit.test import QiskitTestCase
from qiskit.pulse.transforms import compress_pulses


def get_pulse_ids(schedule: Schedule) -> Set[int]:
    """Returns ids of pulses used in Schedule."""
    return {inst.pulse.id for _, inst in schedule.instructions}


class TestCompressTransform(QiskitTestCase):
    """Compress function test."""

    def test_with_duplicates(self):
        """Test compression of schedule."""
        schedule = Schedule()
        drive_channel = DriveChannel(0)
        schedule += Play(SamplePulse([0.0, 0.1]), drive_channel)
        schedule += Play(SamplePulse([0.0, 0.1]), drive_channel)

        compressed_schedule = compress_pulses(schedule)
        original_pulse_ids = get_pulse_ids(schedule)
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)

        self.assertEqual(len(compressed_pulse_ids), 1)
        self.assertEqual(len(original_pulse_ids), 2)
        self.assertTrue(next(iter(compressed_pulse_ids)) in original_pulse_ids)

    def test_sample_pulse_with_clipping(self):
        """Test sample pulses with clipping."""
        schedule = Schedule()
        drive_channel = DriveChannel(0)
        schedule += Play(SamplePulse([0.0, 1.0]), drive_channel)
        schedule += Play(SamplePulse([0.0, 1.001], epsilon=1e-3), drive_channel)
        schedule += Play(SamplePulse([0.0, 1.0000000001]), drive_channel)

        compressed_schedule = compress_pulses(schedule)
        original_pulse_ids = get_pulse_ids(schedule)
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)

        self.assertEqual(len(compressed_pulse_ids), 1)
        self.assertEqual(len(original_pulse_ids), 3)
        self.assertTrue(next(iter(compressed_pulse_ids)) in original_pulse_ids)

    def test_no_duplicates(self):
        """Test with no pulse duplicates."""
        schedule = Schedule()
        drive_channel = DriveChannel(0)
        schedule += Play(SamplePulse([0.0, 1.0]), drive_channel)
        schedule += Play(SamplePulse([0.0, 0.9]), drive_channel)
        schedule += Play(SamplePulse([0.0, 0.3]), drive_channel)

        compressed_schedule = compress_pulses(schedule)
        original_pulse_ids = get_pulse_ids(schedule)
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)
        self.assertEqual(len(original_pulse_ids), len(compressed_pulse_ids))

    def test_parametric_pulses_with_duplicates(self):
        """Test with parametric pulses."""
        schedule = Schedule()
        drive_channel = DriveChannel(0)
        schedule += Play(Gaussian(duration=25, sigma=4, amp=0.5j), drive_channel)
        schedule += Play(Gaussian(duration=25, sigma=4, amp=0.5j), drive_channel)
        schedule += Play(GaussianSquare(duration=150, amp=0.2,
                                        sigma=8, width=140), drive_channel)
        schedule += Play(GaussianSquare(duration=150, amp=0.2,
                                        sigma=8, width=140), drive_channel)
        schedule += Play(ConstantPulse(duration=150, amp=0.1 + 0.4j), drive_channel)
        schedule += Play(ConstantPulse(duration=150, amp=0.1 + 0.4j), drive_channel)
        schedule += Play(Drag(duration=25, amp=0.2 + 0.3j, sigma=7.8, beta=4), drive_channel)
        schedule += Play(Drag(duration=25, amp=0.2 + 0.3j, sigma=7.8, beta=4), drive_channel)

        compressed_schedule = compress_pulses(schedule)
        original_pulse_ids = get_pulse_ids(schedule)
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)
        self.assertEqual(len(original_pulse_ids), 8)
        self.assertEqual(len(compressed_pulse_ids), 4)

    def test_parametric_pulses_with_no_duplicates(self):
        """Test parametric pulses with no duplicates."""
        schedule = Schedule()
        drive_channel = DriveChannel(0)
        schedule += Play(Gaussian(duration=25, sigma=4, amp=0.5j), drive_channel)
        schedule += Play(Gaussian(duration=25, sigma=4, amp=0.49j), drive_channel)
        schedule += Play(GaussianSquare(duration=150, amp=0.2,
                                        sigma=8, width=140), drive_channel)
        schedule += Play(GaussianSquare(duration=150, amp=0.19,
                                        sigma=8, width=140), drive_channel)
        schedule += Play(ConstantPulse(duration=150, amp=0.1 + 0.4j), drive_channel)
        schedule += Play(ConstantPulse(duration=150, amp=0.1 + 0.41j), drive_channel)
        schedule += Play(Drag(duration=25, amp=0.2 + 0.3j, sigma=7.8, beta=4), drive_channel)
        schedule += Play(Drag(duration=25, amp=0.2 + 0.31j, sigma=7.8, beta=4), drive_channel)

        compressed_schedule = compress_pulses(schedule)
        original_pulse_ids = get_pulse_ids(schedule)
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)
        self.assertEqual(len(original_pulse_ids), len(compressed_pulse_ids))

    def test_with_different_channels(self):
        """Test with different channels."""
        schedule = Schedule()
        schedule += Play(SamplePulse([0.0, 0.1]), DriveChannel(0))
        schedule += Play(SamplePulse([0.0, 0.1]), DriveChannel(1))

        compressed_schedule = compress_pulses(schedule)
        original_pulse_ids = get_pulse_ids(schedule)
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)
        self.assertEqual(len(original_pulse_ids), 2)
        self.assertEqual(len(compressed_pulse_ids), 1)

    def test_with_by_channel_compression(self):
        """Test by channel compression."""
        schedule = Schedule()
        schedule += Play(SamplePulse([0.0, 0.1]), DriveChannel(0))
        schedule += Play(SamplePulse([0.0, 0.1]), DriveChannel(0))
        schedule += Play(SamplePulse([0.0, 0.1]), DriveChannel(1))

        compressed_schedule = compress_pulses(schedule, by_channel=True)
        original_pulse_ids = get_pulse_ids(schedule)
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)
        self.assertEqual(len(original_pulse_ids), 3)
        self.assertEqual(len(compressed_pulse_ids), 2)

    def test_sample_pulses_with_tolerance(self):
        """Test sample pulses with tolerance."""
        schedule = Schedule()
        schedule += Play(SamplePulse([0.0, 0.1001], epsilon=1e-3), DriveChannel(0))
        schedule += Play(SamplePulse([0.0, 0.1], epsilon=1e-3), DriveChannel(1))

        compressed_schedule = compress_pulses(schedule)
        original_pulse_ids = get_pulse_ids(schedule)
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)
        self.assertEqual(len(original_pulse_ids), 2)
        self.assertEqual(len(compressed_pulse_ids), 1)
