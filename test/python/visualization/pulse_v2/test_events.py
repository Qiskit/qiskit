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
from qiskit.visualization.pulse_v2 import events


class TestChannelEvents(QiskitTestCase):
    """Tests for ChannelEvents."""
    def test_parse_program(self):
        """Test typical pulse program."""
        test_pulse = pulse.Gaussian(10, 0.1, 3)

        sched = pulse.Schedule()
        sched = sched.insert(0, pulse.SetPhase(3.14, pulse.DriveChannel(0)))
        sched = sched.insert(0, pulse.Play(test_pulse, pulse.DriveChannel(0)))
        sched = sched.insert(10, pulse.ShiftPhase(-1.57, pulse.DriveChannel(0)))
        sched = sched.insert(10, pulse.Play(test_pulse, pulse.DriveChannel(0)))

        ch_events = events.ChannelEvents.load_program(sched, pulse.DriveChannel(0))

        # check waveform data
        waveforms = list(ch_events.get_waveforms())
        inst_data0 = waveforms[0]
        self.assertEqual(inst_data0.t0, 0)
        self.assertEqual(inst_data0.frame.phase, 3.14)
        self.assertEqual(inst_data0.frame.freq, 0)
        self.assertEqual(inst_data0.inst, pulse.Play(test_pulse, pulse.DriveChannel(0)))

        inst_data1 = waveforms[1]
        self.assertEqual(inst_data1.t0, 10)
        self.assertEqual(inst_data1.frame.phase, 1.57)
        self.assertEqual(inst_data1.frame.freq, 0)
        self.assertEqual(inst_data1.inst, pulse.Play(test_pulse, pulse.DriveChannel(0)))

        # check frame data
        frames = list(ch_events.get_frame_changes())
        inst_data0 = frames[0]
        self.assertEqual(inst_data0.t0, 0)
        self.assertEqual(inst_data0.frame.phase, 3.14)
        self.assertEqual(inst_data0.frame.freq, 0)
        self.assertListEqual(inst_data0.inst, [pulse.SetPhase(3.14, pulse.DriveChannel(0))])

        inst_data1 = frames[1]
        self.assertEqual(inst_data1.t0, 10)
        self.assertEqual(inst_data1.frame.phase, -1.57)
        self.assertEqual(inst_data1.frame.freq, 0)
        self.assertListEqual(inst_data1.inst, [pulse.ShiftPhase(-1.57, pulse.DriveChannel(0))])

    def test_multiple_frames_at_the_same_time(self):
        """Test multiple frame instruction at the same time."""
        # shift phase followed by set phase
        sched = pulse.Schedule()
        sched = sched.insert(0, pulse.ShiftPhase(-1.57, pulse.DriveChannel(0)))
        sched = sched.insert(0, pulse.SetPhase(3.14, pulse.DriveChannel(0)))

        ch_events = events.ChannelEvents.load_program(sched, pulse.DriveChannel(0))
        frames = list(ch_events.get_frame_changes())
        inst_data0 = frames[0]
        self.assertAlmostEqual(inst_data0.frame.phase, 3.14)

        # set phase followed by shift phase
        sched = pulse.Schedule()
        sched = sched.insert(0, pulse.SetPhase(3.14, pulse.DriveChannel(0)))
        sched = sched.insert(0, pulse.ShiftPhase(-1.57, pulse.DriveChannel(0)))

        ch_events = events.ChannelEvents.load_program(sched, pulse.DriveChannel(0))
        frames = list(ch_events.get_frame_changes())
        inst_data0 = frames[0]
        self.assertAlmostEqual(inst_data0.frame.phase, 1.57)

    def test_frequency(self):
        """Test parse frequency."""
        sched = pulse.Schedule()
        sched = sched.insert(0, pulse.ShiftFrequency(1.0, pulse.DriveChannel(0)))
        sched = sched.insert(5, pulse.SetFrequency(5.0, pulse.DriveChannel(0)))

        ch_events = events.ChannelEvents.load_program(sched, pulse.DriveChannel(0))
        ch_events.set_config(dt=0.1, init_frequency=3.0, init_phase=0)
        frames = list(ch_events.get_frame_changes())

        inst_data0 = frames[0]
        self.assertAlmostEqual(inst_data0.frame.freq, 1.0)

        inst_data1 = frames[1]
        self.assertAlmostEqual(inst_data1.frame.freq, 1.0)
