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

r"""
Core module of the pulse drawer.

"""

from typing import Union, Optional, Dict, List, Tuple

from qiskit import pulse
from qiskit.visualization.pulse_v2 import events, data_types, drawing_objects, PULSE_STYLE
from qiskit.visualization.exceptions import VisualizationError
from qiskit.providers import BaseBackend


class DrawDataContainer:
    """aaa"""

    _draw_channels = tuple((pulse.DriveChannel,
                            pulse.ControlChannel,
                            pulse.MeasureChannel,
                            pulse.AcquireChannel))

    def __init__(self,
                 dt: Optional[int] = None,
                 drive_los: Optional[Dict[int, float]] = None,
                 control_los: Optional[Dict[int, float]] = None,
                 measure_los: Optional[Dict[int, float]] = None,
                 backend: Optional[BaseBackend] = None):
        """aaa"""

        self.dt = None
        self.d_los = dict()
        self.c_los = dict()
        self.m_los = dict()
        self.channels = set()
        self.drawings = []
        self.chan_event_table = dict()

        # boundary box
        self.bbox_top = 0
        self.bbox_bottom = 0
        self.bbox_left = 0
        self.bbox_right = 0

        # load default settings
        if backend is not None:
            self._load_iqx_backend(backend)

        # overwrite default values
        if drive_los is not None:
            self.d_los = drive_los

        if control_los is not None:
            self.c_los = control_los

        if measure_los is not None:
            self.m_los = measure_los

        if dt is not None:
            self.dt = dt

    def _load_iqx_backend(self,
                          backend: BaseBackend):
        configuration = backend.configuration()
        defaults = backend.defaults()

        self.dt = configuration.dt

        self.d_los = {ind: val for ind, val in enumerate(defaults.qubit_freq_est)}
        self.m_los = {ind: val for ind, val in enumerate(defaults.meas_freq_est)}
        self.c_los = dict()

        for ind, u_lo_mappers in enumerate(configuration.u_channel_lo):
            temp_val = 0
            for u_lo_mapper in u_lo_mappers:
                temp_val = self.d_los[u_lo_mapper.q] * complex(*u_lo_mapper.scale)
            self.c_los[ind] = temp_val.real

    def load_program(self, program: Union[pulse.Waveform, pulse.Schedule]):
        if isinstance(program, pulse.Schedule):
            self._schedule_loader(program)
        elif isinstance(program, pulse.Waveform):
            self._waveform_loader(program)
        else:
            raise VisualizationError('Data type %s is not supported.' % type(program))

    @staticmethod
    def _waveform_loader(program: pulse.Waveform):
        """aaa"""
        return 0

    def _schedule_loader(self, program: pulse.Schedule):
        """aaa"""
        # load program by channel
        for chan in program.channels:
            if isinstance(chan, self._draw_channels):
                chan_event = events.ChannelEvents.load_program(program, chan)
                if isinstance(chan, pulse.DriveChannel):
                    lo_freq = self.d_los.get(chan.index, 0)
                elif isinstance(chan, pulse.ControlChannel):
                    lo_freq = self.c_los.get(chan.index, 0)
                elif isinstance(chan, pulse.MeasureChannel):
                    lo_freq = self.m_los.get(chan.index, 0)
                else:
                    lo_freq = 0
                chan_event.config(self.dt, lo_freq, 0)
                self.chan_event_table[chan] = chan_event
                self.channels.add(chan)

        # update time range
        self.set_time_range(0, program.duration)

        # generate drawing objects
        for chan, chan_event in self.chan_event_table.items():
            # create drawing objects for waveform
            for gen in PULSE_STYLE.style['generator.waveform']:
                for drawing in sum(list(map(gen, chan_event.get_waveforms())), []):
                    self._replace_drawing(drawing)
            # create drawing objects for frame change
            for gen in PULSE_STYLE.style['generator.frame']:
                for drawing in sum(list(map(gen, chan_event.get_frame_changes())), []):
                    self._replace_drawing(drawing)
            # create channel info
            chan_info = data_types.ChannelTuple(chan, 1.0)
            for gen in PULSE_STYLE.style['generator.channel']:
                for drawing in gen(chan_info):
                    self._replace_drawing(drawing)

        # create snapshot
        snapshot_sched = program.filter(instruction_types=[pulse.instructions.Snapshot])
        for t0, inst in snapshot_sched.instructions:
            inst_data = data_types.NonPulseTuple(t0, self.dt, inst)
            for gen in PULSE_STYLE.style['generator.snapshot']:
                for drawing in gen(inst_data):
                    self._replace_drawing(drawing)

        # create barrier
        snapshot_sched = program.filter(instruction_types=[pulse.instructions.RelativeBarrier])
        for t0, inst in snapshot_sched.instructions:
            inst_data = data_types.NonPulseTuple(t0, self.dt, inst)
            for gen in PULSE_STYLE.style['generator.barrier']:
                for drawing in gen(inst_data):
                    self._replace_drawing(drawing)

    def set_time_range(self,
                       t_start: Union[int, float],
                       t_end: Union[int, float]):

        # convert into nearest cycle time
        if isinstance(t_start, float):
            if self.dt is not None:
                t_start = int(t_start / self.dt)
            else:
                raise VisualizationError('Floating valued start time %f seems to be in ',
                                         'units of sec but dt is not specified.' % t_start)
        # convert into nearest cycle time
        if isinstance(t_end, float):
            if self.dt is not None:
                t_end = int(t_end / self.dt)
            else:
                raise VisualizationError('Floating valued end time %f seems to be in ',
                                         'units of sec but dt is not specified.' % t_end)

        duration = t_end - t_start

        self.bbox_left = t_start - int(duration * PULSE_STYLE.style['formatter.margin.left'])
        self.bbox_right = t_end + int(duration * PULSE_STYLE.style['formatter.margin.right'])

    def update_channel_property(self,
                                visible_channels: Optional[List[pulse.channels.Channel]] = None,
                                scales: Optional[Dict[pulse.channels.Channel, float]] = None):
        scales = scales or dict()

        # arrange channels to show
        ordered_channels = self._ordered_channels(visible_channels)

        # new properties
        chan_visible = {chan: False for chan in self.channels}
        chan_offset = {chan: 0.0 for chan in self.channels}
        chan_scale = {chan: 1.0 for chan in self.channels}

        # update channel property
        time_range = (self.bbox_left, self.bbox_right)
        y0 = - PULSE_STYLE.style['formatter.margin.top']
        y0_interval = PULSE_STYLE.style['formatter.margin.between_channel']
        for chan in ordered_channels:

            min_v, max_v = self.chan_event_table[chan].get_min_max(time_range)
            min_v = min(PULSE_STYLE.style['formatter.channel_scaling.min_height'], min_v)
            max_v = max(PULSE_STYLE.style['formatter.channel_scaling.max_height'], max_v)

            # calculate scaling
            if chan in scales:
                # channel is specified by user
                scale = scales[chan]
            elif type(chan) in scales:
                # channel type is specified by user
                scale = scales[type(chan)]
            elif PULSE_STYLE.style['formatter.control.auto_channel_scaling']:
                # auto scaling is enabled
                max_abs_val = max(abs(max_v), abs(min_v))
                scale = 1 / max_abs_val
            else:
                # not specified by user, no auto scale, then apply default scaling
                if isinstance(chan, pulse.DriveChannel):
                    scale = PULSE_STYLE.style['formatter.channel_scaling.drive']
                elif isinstance(chan, pulse.ControlChannel):
                    scale = PULSE_STYLE.style['formatter.channel_scaling.control']
                elif isinstance(chan, pulse.MeasureChannel):
                    scale = PULSE_STYLE.style['formatter.channel_scaling.measure']
                elif isinstance(chan, pulse.AcquireChannel):
                    scale = PULSE_STYLE.style['formatter.channel_scaling.acquire']
                else:
                    scale = 1.0

            # calculate offset coordinate
            offset = y0 - scale * max_v

            # update properties
            chan_visible[chan] = True
            chan_offset[chan] = offset
            chan_scale[chan] = scale

            y0 -= scale * min_v + y0_interval

        # update drawing objects
        for chan in self.channels:
            # update channel info to replace scaling factor
            chan_info = data_types.ChannelTuple(chan, chan_scale.get(chan, 1.0))
            for gen in PULSE_STYLE.style['generator.channel']:
                for drawing in gen(chan_info):
                    self._replace_drawing(drawing)

            # update existing drawings
            for drawing in self.drawings:
                if drawing.channel == chan:
                    drawing.visible = chan_visible.get(chan, False)
                    drawing.offset = chan_offset.get(chan, 0.0)
                    drawing.scale = chan_scale.get(chan, 1.0)

        # update boundary box
        self.bbox_bottom = y0 - (PULSE_STYLE.style['formatter.margin.bottom'] - y0_interval)

    def _ordered_channels(self,
                          visible_channels: Optional[List[pulse.channels.Channel]] = None):

        if visible_channels is None:
            channels = []
            for chan in self.channels:
                # remove acquire
                if PULSE_STYLE.style['formatter.control.show_acquire_channel'] and \
                        isinstance(chan, pulse.AcquireChannel):
                    continue
                # remove empty
                if PULSE_STYLE.style['formatter.control.show_empty_channel'] and \
                        self.chan_event_table[chan].is_empty():
                    continue
                channels.append(chan)
        else:
            channels = visible_channels

        # callback function to arrange channels
        layout_pattern = PULSE_STYLE.style['layout.channel']

        return layout_pattern(channels)

    def _replace_drawing(self,
                         drawing: drawing_objects.ElementaryData):

        if drawing in self.drawings:
            ind = self.drawings.index(drawing)
            self.drawings[ind] = drawing
        else:
            self.drawings.append(drawing)
