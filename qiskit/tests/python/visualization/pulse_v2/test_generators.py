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

"""Tests for drawing object generation."""

import numpy as np

from qiskit import pulse
from qiskit.test import QiskitTestCase
from qiskit.visualization.pulse_v2 import (drawing_objects,
                                           generators,
                                           types)
from qiskit.visualization.pulse_v2.style import stylesheet


class TestGenerators(QiskitTestCase):
    """Tests for generators."""

    def setUp(self) -> None:
        self.style = stylesheet.QiskitPulseStyle()

    @staticmethod
    def create_instruction(inst, phase, freq, t0, dt):
        """A helper function to create InstructionTuple."""
        frame = types.PhaseFreqTuple(phase=phase, freq=freq)
        return types.InstructionTuple(t0=t0, dt=dt, frame=frame, inst=inst)

    def test_gen_filled_waveform_stepwise_play(self):
        """Test gen_filled_waveform_stepwise with play instruction."""
        my_pulse = pulse.Waveform(samples=[0, 0.5+0.5j, 0.5+0.5j, 0], name='my_pulse')
        play = pulse.Play(my_pulse, pulse.DriveChannel(0))
        inst_data = self.create_instruction(play, np.pi/2, 5e9, 5, 0.1)
        objs = generators.gen_filled_waveform_stepwise(inst_data)

        self.assertEqual(len(objs), 2)

        # type check
        self.assertEqual(type(objs[0]), drawing_objects.FilledAreaData)
        self.assertEqual(type(objs[1]), drawing_objects.FilledAreaData)

        y1_ref = np.array([0, 0, -0.5, -0.5, -0.5, -0.5, 0, 0])
        y2_ref = np.array([0, 0, 0, 0, 0, 0, 0, 0])

        # data check
        self.assertEqual(objs[0].channel, pulse.DriveChannel(0))
        self.assertListEqual(list(objs[0].x), [5, 6, 6, 7, 7, 8, 8, 9])
        np.testing.assert_array_almost_equal(objs[0].y1, y1_ref)
        np.testing.assert_array_almost_equal(objs[0].y2, y2_ref)

        # meta data check
        ref_meta = {'duration (cycle time)': 4,
                    'duration (sec)': 0.4,
                    't0 (cycle time)': 5,
                    't0 (sec)': 0.5,
                    'phase': np.pi/2,
                    'frequency': 5e9,
                    'name': 'my_pulse',
                    'data': 'real'}
        self.assertDictEqual(objs[0].meta, ref_meta)

        # style check
        ref_style = {'alpha': self.style['formatter.alpha.fill_waveform'],
                     'zorder': self.style['formatter.layer.fill_waveform'],
                     'linewidth': self.style['formatter.line_width.fill_waveform'],
                     'linestyle': self.style['formatter.line_style.fill_waveform'],
                     'color': self.style['formatter.color.fill_waveform_d'][0]}
        self.assertDictEqual(objs[0].styles, ref_style)

    def test_gen_filled_waveform_stepwise_acquire(self):
        """Test gen_filled_waveform_stepwise with acquire instruction."""
        acquire = pulse.Acquire(duration=4,
                                channel=pulse.AcquireChannel(0),
                                mem_slot=pulse.MemorySlot(0),
                                discriminator=pulse.Discriminator(name='test_discr'),
                                name='acquire')
        inst_data = self.create_instruction(acquire, 0, 7e9, 5, 0.1)

        objs = generators.gen_filled_waveform_stepwise(inst_data)

        # imaginary part is empty and not returned
        self.assertEqual(len(objs), 1)

        # type check
        self.assertEqual(type(objs[0]), drawing_objects.FilledAreaData)

        y1_ref = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        y2_ref = np.array([0, 0, 0, 0, 0, 0, 0, 0])

        # data check
        self.assertEqual(objs[0].channel, pulse.AcquireChannel(0))
        self.assertListEqual(list(objs[0].x), [5, 6, 6, 7, 7, 8, 8, 9])
        np.testing.assert_array_almost_equal(objs[0].y1, y1_ref)
        np.testing.assert_array_almost_equal(objs[0].y2, y2_ref)

        # meta data check
        ref_meta = {'memory slot': 'm0',
                    'register slot': 'N/A',
                    'discriminator': 'test_discr',
                    'kernel': 'N/A',
                    'duration (cycle time)': 4,
                    'duration (sec)': 0.4,
                    't0 (cycle time)': 5,
                    't0 (sec)': 0.5,
                    'phase': 0,
                    'frequency': 7e9,
                    'name': 'acquire',
                    'data': 'real'}

        self.assertDictEqual(objs[0].meta, ref_meta)

        # style check
        ref_style = {'alpha': self.style['formatter.alpha.fill_waveform'],
                     'zorder': self.style['formatter.layer.fill_waveform'],
                     'linewidth': self.style['formatter.line_width.fill_waveform'],
                     'linestyle': self.style['formatter.line_style.fill_waveform'],
                     'color': self.style['formatter.color.fill_waveform_a'][0]}
        self.assertDictEqual(objs[0].styles, ref_style)

    def test_gen_iqx_latex_waveform_name_x90(self):
        """Test gen_iqx_latex_waveform_name with x90 waveform."""
        iqx_pulse = pulse.Waveform(samples=[0, 0, 0, 0], name='X90p_d0_1234567')
        play = pulse.Play(iqx_pulse, pulse.DriveChannel(0))
        inst_data = self.create_instruction(play, 0, 0, 0, 0.1)

        obj = generators.gen_iqx_latex_waveform_name(inst_data)[0]

        # type check
        self.assertEqual(type(obj), drawing_objects.TextData)

        # data check
        self.assertEqual(obj.channel, pulse.DriveChannel(0))
        self.assertEqual(obj.text, 'X90p_d0_1234567')
        self.assertEqual(obj.latex, r'{\rm X}(\frac{\pi}{2})')

        # style check
        ref_style = {'zorder': self.style['formatter.layer.annotate'],
                     'color': self.style['formatter.color.annotate'],
                     'size': self.style['formatter.text_size.annotate'],
                     'va': 'center',
                     'ha': 'center'}
        self.assertDictEqual(obj.styles, ref_style)

    def test_gen_iqx_latex_waveform_name_x180(self):
        """Test gen_iqx_latex_waveform_name with x180 waveform."""
        iqx_pulse = pulse.Waveform(samples=[0, 0, 0, 0], name='Xp_d0_1234567')
        play = pulse.Play(iqx_pulse, pulse.DriveChannel(0))
        inst_data = self.create_instruction(play, 0, 0, 0, 0.1)

        obj = generators.gen_iqx_latex_waveform_name(inst_data)[0]

        # type check
        self.assertEqual(type(obj), drawing_objects.TextData)

        # data check
        self.assertEqual(obj.channel, pulse.DriveChannel(0))
        self.assertEqual(obj.text, 'Xp_d0_1234567')
        self.assertEqual(obj.latex, r'{\rm X}(\pi)')

    def test_gen_iqx_latex_waveform_name_cr(self):
        """Test gen_iqx_latex_waveform_name with CR waveform."""
        iqx_pulse = pulse.Waveform(samples=[0, 0, 0, 0], name='CR90p_u0_1234567')
        play = pulse.Play(iqx_pulse, pulse.ControlChannel(0))
        inst_data = self.create_instruction(play, 0, 0, 0, 0.1)

        obj = generators.gen_iqx_latex_waveform_name(inst_data)[0]

        # type check
        self.assertEqual(type(obj), drawing_objects.TextData)

        # data check
        self.assertEqual(obj.channel, pulse.ControlChannel(0))
        self.assertEqual(obj.text, 'CR90p_u0_1234567')
        self.assertEqual(obj.latex, r'{\rm CR}(\frac{\pi}{4})')

    def test_gen_iqx_latex_waveform_name_compensation_tone(self):
        """Test gen_iqx_latex_waveform_name with CR compensation waveform."""
        iqx_pulse = pulse.Waveform(samples=[0, 0, 0, 0], name='CR90p_d0_u0_1234567')
        play = pulse.Play(iqx_pulse, pulse.DriveChannel(0))
        inst_data = self.create_instruction(play, 0, 0, 0, 0.1)

        obj = generators.gen_iqx_latex_waveform_name(inst_data)[0]

        # type check
        self.assertEqual(type(obj), drawing_objects.TextData)

        # data check
        self.assertEqual(obj.channel, pulse.DriveChannel(0))
        self.assertEqual(obj.text, 'CR90p_d0_u0_1234567')
        self.assertEqual(obj.latex, r'\overline{\rm CR}(\frac{\pi}{4})')

    def test_gen_baseline(self):
        """Test gen_baseline."""
        channel_info = types.ChannelTuple(channel=pulse.DriveChannel(0), scaling=1)

        obj = generators.gen_baseline(channel_info)[0]

        # type check
        self.assertEqual(type(obj), drawing_objects.LineData)

        # data check
        self.assertEqual(obj.channel, pulse.DriveChannel(0))
        self.assertEqual(obj.x, None)
        self.assertEqual(obj.y, 0)

        # style check
        ref_style = {'alpha': self.style['formatter.alpha.baseline'],
                     'zorder': self.style['formatter.layer.baseline'],
                     'linewidth': self.style['formatter.line_width.baseline'],
                     'linestyle': self.style['formatter.line_style.baseline'],
                     'color': self.style['formatter.color.baseline']}
        self.assertDictEqual(obj.styles, ref_style)

    def test_gen_latex_channel_name(self):
        """Test gen_latex_channel_name."""
        channel_info = types.ChannelTuple(channel=pulse.DriveChannel(0), scaling=0.5)

        obj = generators.gen_latex_channel_name(channel_info)[0]

        # type check
        self.assertEqual(type(obj), drawing_objects.TextData)

        # data check
        self.assertEqual(obj.channel, pulse.DriveChannel(0))
        self.assertEqual(obj.latex, 'D_0')
        self.assertEqual(obj.text, 'D0')

        # style check
        ref_style = {'zorder': self.style['formatter.layer.axis_label'],
                     'color': self.style['formatter.color.axis_label'],
                     'size': self.style['formatter.text_size.axis_label'],
                     'va': 'center',
                     'ha': 'right'}
        self.assertDictEqual(obj.styles, ref_style)

    def test_gen_gen_scaling_info(self):
        """Test gen_scaling_info."""
        channel_info = types.ChannelTuple(channel=pulse.DriveChannel(0), scaling=0.5)

        obj = generators.gen_scaling_info(channel_info)[0]

        # type check
        self.assertEqual(type(obj), drawing_objects.TextData)

        # data check
        self.assertEqual(obj.channel, pulse.DriveChannel(0))
        self.assertEqual(obj.text, 'x0.5')

        # style check
        ref_style = {'zorder': self.style['formatter.layer.axis_label'],
                     'color': self.style['formatter.color.axis_label'],
                     'size': self.style['formatter.text_size.annotate'],
                     'va': 'center',
                     'ha': 'right'}
        self.assertDictEqual(obj.styles, ref_style)

    def test_gen_latex_vz_label(self):
        """Test gen_latex_vz_label."""
        fcs = [pulse.ShiftPhase(np.pi/2, pulse.DriveChannel(0)),
               pulse.ShiftFrequency(1e6, pulse.DriveChannel(0))]
        inst_data = self.create_instruction(fcs, np.pi/2, 1e6, 5, 0.1)

        obj = generators.gen_latex_vz_label(inst_data)[0]

        # type check
        self.assertEqual(type(obj), drawing_objects.TextData)

        # data check
        self.assertEqual(obj.channel, pulse.DriveChannel(0))
        self.assertEqual(obj.latex, r'{\rm VZ}(-\frac{\pi}{2})')
        self.assertEqual(obj.text, r'VZ(-1.57 rad.)')

        # style check
        ref_style = {'zorder': self.style['formatter.layer.frame_change'],
                     'color': self.style['formatter.color.frame_change'],
                     'size': self.style['formatter.text_size.annotate'],
                     'va': 'center',
                     'ha': 'center'}
        self.assertDictEqual(obj.styles, ref_style)

    def test_gen_latex_frequency_mhz_value(self):
        """Test gen_latex_frequency_mhz_value."""
        fcs = [pulse.ShiftPhase(np.pi/2, pulse.DriveChannel(0)),
               pulse.ShiftFrequency(1e6, pulse.DriveChannel(0))]
        inst_data = self.create_instruction(fcs, np.pi/2, 1e6, 5, 0.1)

        obj = generators.gen_latex_frequency_mhz_value(inst_data)[0]

        # type check
        self.assertEqual(type(obj), drawing_objects.TextData)

        # data check
        self.assertEqual(obj.channel, pulse.DriveChannel(0))
        self.assertEqual(obj.latex, r'\Delta f = 1.00 ~{\rm MHz}')
        self.assertEqual(obj.text, u'\u0394' + 'f=1.00 MHz')

        # style check
        ref_style = {'zorder': self.style['formatter.layer.frame_change'],
                     'color': self.style['formatter.color.frame_change'],
                     'size': self.style['formatter.text_size.annotate'],
                     'va': 'center',
                     'ha': 'center'}
        self.assertDictEqual(obj.styles, ref_style)

    def test_gen_raw_frame_operand_values(self):
        """Test gen_raw_frame_operand_values."""
        fcs = [pulse.ShiftPhase(np.pi/2, pulse.DriveChannel(0)),
               pulse.ShiftFrequency(1e6, pulse.DriveChannel(0))]
        inst_data = self.create_instruction(fcs, np.pi/2, 1e6, 5, 0.1)

        obj = generators.gen_raw_frame_operand_values(inst_data)[0]

        # type check
        self.assertEqual(type(obj), drawing_objects.TextData)

        # data check
        self.assertEqual(obj.channel, pulse.DriveChannel(0))
        self.assertEqual(obj.text, r'(1.57, 1.0e+06)')

        # style check
        ref_style = {'zorder': self.style['formatter.layer.frame_change'],
                     'color': self.style['formatter.color.frame_change'],
                     'size': self.style['formatter.text_size.annotate'],
                     'va': 'center',
                     'ha': 'center'}
        self.assertDictEqual(obj.styles, ref_style)

    def test_gen_frame_symbol(self):
        """Test gen_frame_symbol."""
        fcs = [pulse.ShiftPhase(np.pi/2, pulse.DriveChannel(0)),
               pulse.ShiftFrequency(1e6, pulse.DriveChannel(0))]
        inst_data = self.create_instruction(fcs, np.pi/2, 1e6, 5, 0.1)

        obj = generators.gen_frame_symbol(inst_data)[0]

        # type check
        self.assertEqual(type(obj), drawing_objects.TextData)

        # data check
        self.assertEqual(obj.channel, pulse.DriveChannel(0))
        self.assertEqual(obj.latex, self.style['formatter.latex_symbol.frame_change'])
        self.assertEqual(obj.text, self.style['formatter.unicode_symbol.frame_change'])

        # metadata check
        ref_meta = {
            'total phase change': np.pi/2,
            'total frequency change': 1e6,
            'program': ['ShiftPhase(1.57 rad.)', 'ShiftFrequency(1.00e+06 Hz)'],
            't0 (cycle time)': 5,
            't0 (sec)': 0.5
        }
        self.assertDictEqual(obj.meta, ref_meta)

        # style check
        ref_style = {'zorder': self.style['formatter.layer.frame_change'],
                     'color': self.style['formatter.color.frame_change'],
                     'size': self.style['formatter.text_size.frame_change'],
                     'va': 'center',
                     'ha': 'center'}
        self.assertDictEqual(obj.styles, ref_style)

    def test_gen_snapshot_symbol(self):
        """Test gen_snapshot_symbol."""
        snapshot = pulse.instructions.Snapshot(label='test_snapshot', snapshot_type='statevector')
        inst_data = types.NonPulseTuple(5, 0.1, snapshot)
        symbol, label = generators.gen_snapshot_symbol(inst_data)

        # type check
        self.assertEqual(type(symbol), drawing_objects.TextData)
        self.assertEqual(type(label), drawing_objects.TextData)

        # data check
        self.assertEqual(symbol.channel, pulse.channels.SnapshotChannel())
        self.assertEqual(symbol.text, self.style['formatter.unicode_symbol.snapshot'])
        self.assertEqual(symbol.latex, self.style['formatter.latex_symbol.snapshot'])

        self.assertEqual(label.channel, pulse.channels.SnapshotChannel())
        self.assertEqual(label.text, 'test_snapshot')

        # metadata check
        ref_meta = {'snapshot type': 'statevector',
                    't0 (cycle time)': 5,
                    't0 (sec)': 0.5}
        self.assertDictEqual(symbol.meta, ref_meta)

        # style check
        ref_style = {'zorder': self.style['formatter.layer.snapshot'],
                     'color': self.style['formatter.color.snapshot'],
                     'size': self.style['formatter.text_size.snapshot'],
                     'va': 'bottom',
                     'ha': 'center'}
        self.assertDictEqual(symbol.styles, ref_style)

        ref_style = {'zorder': self.style['formatter.layer.snapshot'],
                     'color': self.style['formatter.color.snapshot'],
                     'size': self.style['formatter.text_size.annotate'],
                     'va': 'bottom',
                     'ha': 'center'}
        self.assertDictEqual(label.styles, ref_style)

    def test_gen_barrier(self):
        """Test gen_barrier."""
        barrier = pulse.instructions.RelativeBarrier(pulse.DriveChannel(0),
                                                     pulse.ControlChannel(0))
        inst_data = types.NonPulseTuple(5, 0.1, barrier)
        lines = generators.gen_barrier(inst_data)

        self.assertEqual(len(lines), 2)

        # type check
        self.assertEqual(type(lines[0]), drawing_objects.LineData)
        self.assertEqual(type(lines[1]), drawing_objects.LineData)

        # data check
        self.assertEqual(lines[0].channel, pulse.channels.DriveChannel(0))
        self.assertEqual(lines[1].channel, pulse.channels.ControlChannel(0))
        self.assertEqual(lines[0].x, 5)
        self.assertEqual(lines[0].y, None)

        # style check
        ref_style = {'alpha': self.style['formatter.alpha.barrier'],
                     'zorder': self.style['formatter.layer.barrier'],
                     'linewidth': self.style['formatter.line_width.barrier'],
                     'linestyle': self.style['formatter.line_style.barrier'],
                     'color': self.style['formatter.color.barrier']}
        self.assertDictEqual(lines[0].styles, ref_style)
