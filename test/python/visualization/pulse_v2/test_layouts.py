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

# pylint: disable=missing-docstring, invalid-name

"""Tests for core modules of pulse drawer."""

from qiskit import pulse
from qiskit.test import QiskitTestCase
from qiskit.visualization.pulse_v2 import layouts


class TestLayout(QiskitTestCase):
    """Tests for layout generation functions."""

    def setUp(self) -> None:
        self.channels = [pulse.DriveChannel(0),
                         pulse.DriveChannel(1),
                         pulse.DriveChannel(2),
                         pulse.MeasureChannel(1),
                         pulse.MeasureChannel(2),
                         pulse.AcquireChannel(1),
                         pulse.AcquireChannel(2),
                         pulse.ControlChannel(0),
                         pulse.ControlChannel(2),
                         pulse.ControlChannel(5)]

    def test_channel_type_grouped_sort(self):
        """Test channel_type_grouped_sort."""
        channels = layouts.channel_type_grouped_sort(self.channels)

        ref_channels = [pulse.DriveChannel(0),
                        pulse.DriveChannel(1),
                        pulse.DriveChannel(2),
                        pulse.ControlChannel(0),
                        pulse.ControlChannel(2),
                        pulse.ControlChannel(5),
                        pulse.MeasureChannel(1),
                        pulse.MeasureChannel(2),
                        pulse.AcquireChannel(1),
                        pulse.AcquireChannel(2)]

        self.assertListEqual(channels, ref_channels)

    def test_channel_index_sort(self):
        """Test channel_index_sort."""
        channels = layouts.channel_index_sort(self.channels)

        ref_channels = [pulse.DriveChannel(0),
                        pulse.ControlChannel(0),
                        pulse.DriveChannel(1),
                        pulse.MeasureChannel(1),
                        pulse.AcquireChannel(1),
                        pulse.DriveChannel(2),
                        pulse.ControlChannel(2),
                        pulse.MeasureChannel(2),
                        pulse.AcquireChannel(2),
                        pulse.ControlChannel(5)]

        self.assertListEqual(channels, ref_channels)

    def test_channel_index_sort_grouped_control(self):
        """Test channel_index_sort_grouped_control."""
        channels = layouts.channel_index_sort_wo_control(self.channels)

        ref_channels = [pulse.DriveChannel(0),
                        pulse.DriveChannel(1),
                        pulse.MeasureChannel(1),
                        pulse.AcquireChannel(1),
                        pulse.DriveChannel(2),
                        pulse.MeasureChannel(2),
                        pulse.AcquireChannel(2),
                        pulse.ControlChannel(0),
                        pulse.ControlChannel(2),
                        pulse.ControlChannel(5)]

        self.assertListEqual(channels, ref_channels)
