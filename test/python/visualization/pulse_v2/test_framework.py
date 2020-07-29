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

import numpy as np

from qiskit import pulse
from qiskit.test import QiskitTestCase
from qiskit.visualization.pulse_v2 import events, core, generators, layouts, PULSE_STYLE
from qiskit.visualization.pulse_v2.style import stylesheet


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

    def test_empty(self):
        """Test is_empty check."""
        test_pulse = pulse.Gaussian(10, 0.1, 3)

        sched = pulse.Schedule()
        sched = sched.insert(0, pulse.ShiftPhase(1.57, pulse.DriveChannel(0)))

        ch_events = events.ChannelEvents.load_program(sched, pulse.DriveChannel(0))
        self.assertTrue(ch_events.is_empty())

        sched = pulse.Schedule()
        sched = sched.insert(0, pulse.Play(test_pulse, pulse.DriveChannel(0)))

        ch_events = events.ChannelEvents.load_program(sched, pulse.DriveChannel(0))
        self.assertFalse(ch_events.is_empty())

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
        ch_events.config(dt=0.1, init_frequency=3.0, init_phase=0)
        frames = list(ch_events.get_frame_changes())

        inst_data0 = frames[0]
        self.assertAlmostEqual(inst_data0.frame.freq, 1.0)

        inst_data1 = frames[1]
        self.assertAlmostEqual(inst_data1.frame.freq, 1.0)

    def test_min_max(self):
        """Test get min max value of channel."""
        test_pulse = pulse.Gaussian(10, 0.1, 3)

        sched = pulse.Schedule()
        sched = sched.insert(0, pulse.Play(test_pulse, pulse.DriveChannel(0)))

        ch_events = events.ChannelEvents.load_program(sched, pulse.DriveChannel(0))

        min_v, max_v = ch_events.get_min_max((0, sched.duration))

        samples = test_pulse.get_sample_pulse().samples

        self.assertAlmostEqual(min_v, min(*samples.real, *samples.imag))
        self.assertAlmostEqual(max_v, max(*samples.real, *samples.imag))


class TestStylesheet(QiskitTestCase):
    """Tests for stylesheet."""
    def test_deprecated_key(self):
        """Test deprecation warning."""
        style = stylesheet.QiskitPulseStyle()
        style._deprecated_keys = {'deprecated_key': 'new_key'}

        with self.assertWarns(DeprecationWarning):
            dep_dict = {
                'deprecated_key': 'value_1'
            }
            style.update(dep_dict)

        self.assertEqual(style['new_key'], 'value_1')


class TestDrawDataContainer(QiskitTestCase):
    """Tests for draw data container."""

    def setUp(self) -> None:
        # draw only waveform, fc symbol, channel name, scaling, baseline, snapshot and barrier
        callbacks_for_test = {'generator.waveform': [generators.gen_filled_waveform_stepwise],
                              'generator.frame': [generators.gen_frame_symbol],
                              'generator.channel': [generators.gen_latex_channel_name,
                                                    generators.gen_scaling_info,
                                                    generators.gen_baseline],
                              'generator.snapshot': [generators.gen_snapshot_symbol],
                              'generator.barrier': [generators.gen_barrier],
                              'layout.channel': layouts.channel_index_sort_wo_control}
        PULSE_STYLE.update(callbacks_for_test)

        gaussian = pulse.Gaussian(40, 0.3, 10)
        square = pulse.Constant(100, 0.2)

        self.sched = pulse.Schedule()
        self.sched = self.sched.insert(0, pulse.Play(pulse=gaussian,
                                                     channel=pulse.DriveChannel(0)))
        self.sched = self.sched.insert(0, pulse.ShiftPhase(phase=np.pi/2,
                                                           channel=pulse.DriveChannel(0)))
        self.sched = self.sched.insert(50, pulse.Play(pulse=square,
                                                      channel=pulse.MeasureChannel(0)))
        self.sched = self.sched.insert(50, pulse.Acquire(duration=100,
                                                         channel=pulse.AcquireChannel(0),
                                                         mem_slot=pulse.MemorySlot(0)))

    def test_loading_backend(self):
        """Test loading backend."""
        from qiskit.test.mock import FakeAthens

        config = FakeAthens().configuration()
        defaults = FakeAthens().defaults()

        ddc = core.DrawDataContainer(backend=FakeAthens())

        # check dt
        self.assertEqual(ddc.dt, config.dt)

        # check drive los
        self.assertEqual(ddc.d_los[0], defaults.qubit_freq_est[0])

        # check measure los
        self.assertEqual(ddc.m_los[0], defaults.meas_freq_est[0])

        # check control los
        self.assertEqual(ddc.c_los[0], defaults.qubit_freq_est[1])

    def test_simple_sched_loading(self):
        """Test data generation with simple schedule."""

        ddc = core.DrawDataContainer()
        ddc.load_program(self.sched)

        # 4 waveform shapes (re, im of gaussian, re of square, re of acquire)
        # 3 channel names
        # 1 fc symbol
        # 3 baselines
        self.assertEqual(len(ddc.drawings), 11)

    def test_simple_sched_reloading(self):
        """Test reloading of the same schedule."""
        ddc = core.DrawDataContainer()
        ddc.load_program(self.sched)

        # the same data should be overwritten
        list_drawing1 = ddc.drawings.copy()
        list_drawing2 = ddc.drawings.copy()

        self.assertListEqual(list_drawing1, list_drawing2)

    def test_update_channels(self):
        """Test update channels."""
        ddc = core.DrawDataContainer()
        ddc.load_program(self.sched)

        ddc.update_channel_property()

        # 2 scale factors are added for d channel and m channel
        self.assertEqual(len(ddc.drawings), 13)

        d_scale = 1 / 0.3
        m_scale = 1 / 0.2
        a_scale = 1

        top_margin = PULSE_STYLE['formatter.margin.top']
        interval = PULSE_STYLE['formatter.margin.between_channel']
        min_h = np.abs(PULSE_STYLE['formatter.channel_scaling.min_height'])

        d_offset = - (top_margin + 1)
        m_offset = - (top_margin + 1 + min_h + interval + 1)
        a_offset = - (top_margin + 1 + min_h + interval + 1 + min_h + interval + 1)

        # check if auto scale factor is correct
        for drawing in ddc.drawings:
            if drawing.channel == pulse.DriveChannel(0):
                self.assertAlmostEqual(drawing.scale, d_scale, places=1)
                self.assertAlmostEqual(drawing.offset, d_offset, places=1)
            elif drawing.channel == pulse.MeasureChannel(0):
                self.assertAlmostEqual(drawing.scale, m_scale, places=1)
                self.assertAlmostEqual(drawing.offset, m_offset, places=1)
            elif drawing.channel == pulse.AcquireChannel(0):
                self.assertAlmostEqual(drawing.scale, a_scale, places=1)
                self.assertAlmostEqual(drawing.offset, a_offset, places=1)

    def test_update_channels_only_drive_channel(self):
        """Test update channels with filtered channels."""
        ddc = core.DrawDataContainer()
        ddc.load_program(self.sched)

        # update
        ddc.update_channel_property(visible_channels=[pulse.DriveChannel(0)])

        # 1 scale factor is added for d channel
        self.assertEqual(len(ddc.drawings), 12)

        # check if visible is updated
        for drawing in ddc.drawings:
            if drawing.channel == pulse.DriveChannel(0):
                self.assertTrue(drawing.visible)
            else:
                self.assertFalse(drawing.visible)

    def test_snapshot(self):
        """Test snapshot instructions."""
        ddc = core.DrawDataContainer()

        sched = pulse.Schedule()
        sched = sched.insert(0, pulse.Snapshot(label='test'))

        ddc.load_program(sched)

        # only snapshot symbol and label text
        self.assertEqual(len(ddc.drawings), 2)

    def test_relative_barrier(self):
        """Test relative barrier instructions."""
        ddc = core.DrawDataContainer()

        sched = pulse.Schedule()
        sched = sched.insert(0, pulse.instructions.RelativeBarrier(pulse.DriveChannel(0)))

        ddc.load_program(sched)

        # barrier line, baseline, channel name
        self.assertEqual(len(ddc.drawings), 3)

    def test_load_waveform(self):
        """Test loading waveform."""
        ddc = core.DrawDataContainer()

        waveform = pulse.library.Constant(duration=10, amp=0.1+0.1j)
        ddc.load_program(waveform)

        # baseline and waveform for real and imaginary
        self.assertEqual(len(ddc.drawings), 3)

    def test_very_long_pulse(self):
        """Test truncation of long pulse."""
        ddc = core.DrawDataContainer()

        very_long_pulse = pulse.Constant(10000, 0.1)
        sched = pulse.Schedule()
        sched = sched.insert(0, pulse.Play(very_long_pulse, pulse.DriveChannel(0)))

        ddc.load_program(sched)
        ddc.update_channel_property()

        self.assertEqual(len(ddc.axis_break), 1)

        removed_t0, removed_t1 = ddc.axis_break[0]
        self.assertEqual(removed_t0, 500)
        self.assertEqual(removed_t1, 9500)
