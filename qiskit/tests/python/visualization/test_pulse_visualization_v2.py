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

# pylint: disable=missing-docstring

"""Tests for IR generation of pulse visualization."""

import numpy as np

from qiskit import pulse
from qiskit.test import QiskitTestCase
from qiskit.visualization.pulse_v2 import drawing_objects
from qiskit.visualization.pulse_v2.events import ChannelEvents


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

        events = ChannelEvents.load_program(sched, pulse.DriveChannel(0))

        # check waveform data
        waveforms = list(events.get_waveforms())
        t0, frame, inst = waveforms[0]
        self.assertEqual(t0, 0)
        self.assertEqual(frame.phase, 3.14)
        self.assertEqual(frame.freq, 0)
        self.assertEqual(inst, pulse.Play(test_pulse, pulse.DriveChannel(0)))

        t0, frame, inst = waveforms[1]
        self.assertEqual(t0, 10)
        self.assertEqual(frame.phase, 1.57)
        self.assertEqual(frame.freq, 0)
        self.assertEqual(inst, pulse.Play(test_pulse, pulse.DriveChannel(0)))

        # check frame data
        frames = list(events.get_frame_changes())
        t0, frame, insts = frames[0]
        self.assertEqual(t0, 0)
        self.assertEqual(frame.phase, 3.14)
        self.assertEqual(frame.freq, 0)
        self.assertListEqual(insts, [pulse.SetPhase(3.14, pulse.DriveChannel(0))])

        t0, frame, insts = frames[1]
        self.assertEqual(t0, 10)
        self.assertEqual(frame.phase, -1.57)
        self.assertEqual(frame.freq, 0)
        self.assertListEqual(insts, [pulse.ShiftPhase(-1.57, pulse.DriveChannel(0))])

    def test_empty(self):
        """Test is_empty check."""
        test_pulse = pulse.Gaussian(10, 0.1, 3)

        sched = pulse.Schedule()
        sched = sched.insert(0, pulse.ShiftPhase(1.57, pulse.DriveChannel(0)))

        events = ChannelEvents.load_program(sched, pulse.DriveChannel(0))
        self.assertTrue(events.is_empty())

        sched = pulse.Schedule()
        sched = sched.insert(0, pulse.Play(test_pulse, pulse.DriveChannel(0)))

        events = ChannelEvents.load_program(sched, pulse.DriveChannel(0))
        self.assertFalse(events.is_empty())

    def test_multiple_frames_at_the_same_time(self):
        """Test multiple frame instruction at the same time."""
        # shift phase followed by set phase
        sched = pulse.Schedule()
        sched = sched.insert(0, pulse.ShiftPhase(-1.57, pulse.DriveChannel(0)))
        sched = sched.insert(0, pulse.SetPhase(3.14, pulse.DriveChannel(0)))

        events = ChannelEvents.load_program(sched, pulse.DriveChannel(0))
        frames = list(events.get_frame_changes())
        _, frame, _ = frames[0]
        self.assertAlmostEqual(frame.phase, 3.14)

        # set phase followed by shift phase
        sched = pulse.Schedule()
        sched = sched.insert(0, pulse.SetPhase(3.14, pulse.DriveChannel(0)))
        sched = sched.insert(0, pulse.ShiftPhase(-1.57, pulse.DriveChannel(0)))

        events = ChannelEvents.load_program(sched, pulse.DriveChannel(0))
        frames = list(events.get_frame_changes())
        _, frame, _ = frames[0]
        self.assertAlmostEqual(frame.phase, 1.57)

    def test_frequency(self):
        """Test parse frequency."""
        sched = pulse.Schedule()
        sched = sched.insert(0, pulse.ShiftFrequency(1.0, pulse.DriveChannel(0)))
        sched = sched.insert(5, pulse.SetFrequency(5.0, pulse.DriveChannel(0)))

        events = ChannelEvents.load_program(sched, pulse.DriveChannel(0))
        events.init_frequency = 3.0
        frames = list(events.get_frame_changes())

        _, frame, _ = frames[0]
        self.assertAlmostEqual(frame.freq, 1.0)

        _, frame, _ = frames[1]
        self.assertAlmostEqual(frame.freq, 1.0)


class TestDrawingObjects(QiskitTestCase):
    """Tests for DrawingObjects."""
    def test_filled_area_data(self):
        """Test FilledAreaData."""
        data1 = drawing_objects.FilledAreaData(data_type='waveform',
                                               channel=pulse.DriveChannel(0),
                                               x=np.array([0, 1, 2]),
                                               y1=np.array([3, 4, 5]),
                                               y2=np.array([0, 0, 0]),
                                               meta={'test_val': 0},
                                               offset=0,
                                               visible=True,
                                               styles={'color': 'red'})

        data2 = drawing_objects.FilledAreaData(data_type='waveform',
                                               channel=pulse.DriveChannel(0),
                                               x=np.array([0, 1, 2]),
                                               y1=np.array([3, 4, 5]),
                                               y2=np.array([0, 0, 0]),
                                               meta={'test_val': 1},
                                               offset=1,
                                               visible=False,
                                               styles={'color': 'blue'})

        self.assertEqual(data1, data2)

    def test_line_data(self):
        """Test for LineData."""
        data1 = drawing_objects.LineData(data_type='baseline',
                                         channel=pulse.DriveChannel(0),
                                         x=np.array([0, 1, 2]),
                                         y=np.array([0, 0, 0]),
                                         meta={'test_val': 0},
                                         offset=0,
                                         visible=True,
                                         styles={'color': 'red'})

        data2 = drawing_objects.LineData(data_type='baseline',
                                         channel=pulse.DriveChannel(0),
                                         x=np.array([0, 1, 2]),
                                         y=np.array([0, 0, 0]),
                                         meta={'test_val': 1},
                                         offset=1,
                                         visible=False,
                                         styles={'color': 'blue'})

        self.assertEqual(data1, data2)

    def test_text_data(self):
        """Test for TextData."""
        data1 = drawing_objects.TextData(data_type='pulse_label',
                                         channel=pulse.DriveChannel(0),
                                         x=0,
                                         y=0,
                                         text='my_text',
                                         meta={'test_val': 0},
                                         offset=0,
                                         visible=True,
                                         styles={'color': 'red'})

        data2 = drawing_objects.TextData(data_type='pulse_label',
                                         channel=pulse.DriveChannel(0),
                                         x=0,
                                         y=0,
                                         text='my_text',
                                         meta={'test_val': 1},
                                         offset=1,
                                         visible=False,
                                         styles={'color': 'blue'})

        self.assertEqual(data1, data2)
