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

# pylint: disable=invalid-name

r"""
Core module of the pulse drawer.

This module provides the `DrawDataContainer` which is a collection of drawing objects
with additional information such as the modulation frequency and the time resolution.
In addition, this instance performs the simple data processing such as channel arrangement,
auto scaling of channels, and truncation of long pulses when a program is loaded.

This class may be initialized with backend instance which plays the schedule,
then a program is loaded and channel information is updated according to the preference:

    ```python
    ddc = DrawDataContainer(backend)
    ddc.load_program(sched)
    ddc.update_channel_property(visible_channels=[DriveChannel(0), DriveChannel(1)])
    ```

If the `DrawDataContainer` is initialized without backend information, the output shows
the time in units of system cycle time `dt` and the frequencies are initialized to zero.

This module is expected to be used by the pulse drawer interface and not exposed to users.

The `DrawDataContainer` takes a schedule of pulse waveform data and converts it into
a set of drawing objects, then a plotter interface takes the drawing objects
from the container to call the plotter's API. The visualization of drawing objects can be
customized with the stylesheet. The generated drawing objects can be accessed from

    ```python
    ddc.drawings
    ```

This module can be commonly used among different plotters. If the plotter supports
dynamic update of drawings, the channel data can be updated with new preference:

    ```python
    ddc.update_channel_property(visible_channels=[DriveChannel(0)])
    ```
In this example, `DriveChannel(1)` will be removed from the output.
"""

from typing import Union, Optional, Dict, List

from qiskit import pulse
from qiskit.providers import BaseBackend
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import events, types, drawing_objects, PULSE_STYLE


class DrawDataContainer:
    """Data container for drawing objects."""

    DEFAULT_DRAW_CHANNELS = tuple((pulse.DriveChannel,
                            pulse.ControlChannel,
                            pulse.MeasureChannel,
                            pulse.AcquireChannel))

    def __init__(self,
                 dt: Optional[int] = None,
                 drive_los: Optional[Dict[int, float]] = None,
                 control_los: Optional[Dict[int, float]] = None,
                 measure_los: Optional[Dict[int, float]] = None,
                 backend: Optional[BaseBackend] = None):
        """Create new data container with backend system information.

        Args:
            dt: Time resolution of this system in units of sec. If this is provided along
                with the `backend`, the extracted property is overwritten by this input.
            drive_los: Dictionary of local oscillator (modulation) frequencies
                of drive channels. If this is provided along with the `backend`,
                the extracted property is overwritten by this input.
            control_los: Dictionary of local oscillator (modulation) frequencies
                of control channels. If this is provided along with the `backend`,
                the extracted property is overwritten by this input.
            measure_los: Dictionary of local oscillator (modulation) frequencies
                of measure channels. If this is provided along with the `backend`,
                the extracted property is overwritten by this input.
            backend: Backend object to play the schedule. If this is provided,
                the time resolution and frequencies are automatically extracted.
        """

        self.dt = None
        self.d_los = dict()
        self.c_los = dict()
        self.m_los = dict()
        self.channels = set()
        self.active_channels = []
        self.chan_event_table = dict()
        self.axis_breaks = []

        # drawing objects
        self.drawings = []

        # boundary box
        self.bbox_top = 0
        self.bbox_bottom = 0
        self.bbox_left = 0
        self.bbox_right = 0

        # load default settings
        if backend:
            self._load_iqx_backend(backend)

        # overwrite default values
        if drive_los:
            self.d_los.update(drive_los)

        if control_los:
            self.c_los.update(control_los)

        if measure_los:
            self.m_los.update(measure_los)

        if dt is not None:
            self.dt = dt

    def _load_iqx_backend(self,
                          backend: BaseBackend):
        """A helper function to extract system property from IQX backend instance.

        Notes:
            The modulation frequencies of control channels should be defined in terms of
            the modulation frequencies of drive channels. This is the syntax of
            IQX backends. If the backend is provided by a third party provider,
            this function may crash or may return wrong frequency values.

        Args:
            backend: Backend object to play the schedule.
        """
        configuration = backend.configuration()
        defaults = backend.defaults()

        self.dt = configuration.dt

        self.d_los = dict(enumerate(defaults.qubit_freq_est))
        self.m_los = dict(enumerate(defaults.meas_freq_est))
        self.c_los = dict()

        for ind, u_lo_mappers in enumerate(configuration.u_channel_lo):
            temp_val = 0
            for u_lo_mapper in u_lo_mappers:
                temp_val = self.d_los[u_lo_mapper.q] * complex(*u_lo_mapper.scale)
            self.c_los[ind] = temp_val.real

    def load_program(self, program: Union[pulse.Waveform, pulse.ParametricPulse, pulse.Schedule]):
        """Load a program to draw.

        Args:
            program: `Waveform`, `ParametricPulse`, or `Schedule` to draw.

        Raises:
            VisualizationError: When input program is invalid data format.
        """
        if isinstance(program, pulse.Schedule):
            self._schedule_loader(program)
        elif isinstance(program, (pulse.Waveform, pulse.ParametricPulse)):
            self._waveform_loader(program)
        else:
            raise VisualizationError('Data type %s is not supported.' % type(program))

    def _waveform_loader(self, program: Union[pulse.Waveform, pulse.ParametricPulse]):
        """Load Waveform instance.

        This function is sub-routine of py:method:`load_program`.

        Args:
            program: `Waveform` to draw.
        """
        sample_channel = types.WaveformChannel()
        inst_tuple = types.InstructionTuple(t0=0,
                                            dt=self.dt,
                                            frame=types.PhaseFreqTuple(phase=0, freq=0),
                                            inst=pulse.Play(program, sample_channel))

        self.set_time_range(0, program.duration)

        # generate waveform related elements
        for gen in PULSE_STYLE['generator.waveform']:
            for drawing in gen(inst_tuple):
                self._replace_drawing(drawing)

        # baseline
        style = {'alpha': PULSE_STYLE['formatter.alpha.baseline'],
                 'zorder': PULSE_STYLE['formatter.layer.baseline'],
                 'linewidth': PULSE_STYLE['formatter.line_width.baseline'],
                 'linestyle': PULSE_STYLE['formatter.line_style.baseline'],
                 'color': PULSE_STYLE['formatter.color.baseline']}

        bline = drawing_objects.LineData(data_type=types.DrawingLine.BASELINE,
                                         channel=sample_channel,
                                         x=[types.AbstractCoordinate.LEFT,
                                            types.AbstractCoordinate.RIGHT],
                                         y=[0, 0],
                                         styles=style)
        self._replace_drawing(bline)

        pulse_data = program if isinstance(program, pulse.Waveform) else program.get_waveform()
        max_v = max(*pulse_data.samples.real, *pulse_data.samples.imag)
        min_v = min(*pulse_data.samples.real, *pulse_data.samples.imag)

        # calculate offset coordinate
        offset = -PULSE_STYLE['formatter.margin.top'] - max_v

        # calculate scaling
        max_abs_val = max(abs(max_v), abs(min_v))
        if max_abs_val < PULSE_STYLE['formatter.general.vertical_resolution'] * 100:
            scale = 1.0
        else:
            scale = 1 / max_abs_val

        for drawing in self.drawings:
            drawing.visible = True
            drawing.offset = offset
            drawing.scale = scale

        # update boundary box
        self.bbox_bottom = offset - min_v - (PULSE_STYLE['formatter.margin.bottom'])

    def _schedule_loader(self, program: pulse.Schedule):
        """Load Schedule instance.

        This function is sub-routine of py:method:`load_program`.

        Args:
            program: `Schedule` to draw.
        """
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
            for gen in PULSE_STYLE['generator.waveform']:
                for drawing in sum(list(map(gen, chan_event.get_waveforms())), []):
                    self._replace_drawing(drawing)
            # create drawing objects for frame change
            for gen in PULSE_STYLE['generator.frame']:
                for drawing in sum(list(map(gen, chan_event.get_frame_changes())), []):
                    self._replace_drawing(drawing)
            # create channel info
            chan_info = types.ChannelTuple(chan, 1.0)
            for gen in PULSE_STYLE['generator.channel']:
                for drawing in gen(chan_info):
                    self._replace_drawing(drawing)

        # create snapshot
        snapshot_sched = program.filter(instruction_types=[pulse.instructions.Snapshot])
        for t0, inst in snapshot_sched.instructions:
            inst_data = types.NonPulseTuple(t0, self.dt, inst)
            for gen in PULSE_STYLE['generator.snapshot']:
                for drawing in gen(inst_data):
                    self._replace_drawing(drawing)

        # create barrier
        snapshot_sched = program.filter(instruction_types=[pulse.instructions.RelativeBarrier])
        for t0, inst in snapshot_sched.instructions:
            inst_data = types.NonPulseTuple(t0, self.dt, inst)
            for gen in PULSE_STYLE['generator.barrier']:
                for drawing in gen(inst_data):
                    self._replace_drawing(drawing)

    def set_time_range(self,
                       t_start: Union[int, float],
                       t_end: Union[int, float]):
        """Set time range to draw.

        The update to time range is applied after :py:method:`update_channel_property` is called.

        Args:
            t_start: Left boundary of drawing in units of cycle time or real time.
            t_end: Right boundary of drawing in units of cycle time or real time.

        Raises:
            VisualizationError: When times are given in float without specifying dt.
        """
        # convert into nearest cycle time
        if isinstance(t_start, float):
            if self.dt is not None:
                t_start = int(t_start / self.dt)
            else:
                raise VisualizationError('Floating valued start time %f seems to be in '
                                         'units of sec but dt is not specified.' % t_start)
        # convert into nearest cycle time
        if isinstance(t_end, float):
            if self.dt is not None:
                t_end = int(t_end / self.dt)
            else:
                raise VisualizationError('Floating valued end time %f seems to be in '
                                         'units of sec but dt is not specified.' % t_end)

        duration = t_end - t_start

        self.bbox_left = t_start - int(duration * PULSE_STYLE['formatter.margin.left'])
        self.bbox_right = t_end + int(duration * PULSE_STYLE['formatter.margin.right'])

    def update_channel_property(self,
                                visible_channels: Optional[List[pulse.channels.Channel]] = None,
                                scales: Optional[Dict[pulse.channels.Channel, float]] = None):
        """Update channel properties.

        This function updates the visibility and scaling of each channel.
        Drawing objects generated by `generator.channel` is regenerated and replaced
        according to the input channel preferences.
        The `visible`, `scale` and `offset` attribute of each drawing object is also updated.

        This function enables a plotter to dynamically update appearance of output image.

        Args:
            visible_channels: List of channels to show.
            scales: Dictionary of scaling factor of channels.
        """
        scales = scales or dict()

        # arrange channels to show
        self.active_channels = self._ordered_channels(visible_channels)

        # new properties
        chan_visible = {chan: False for chan in self.channels}
        chan_offset = {chan: 0.0 for chan in self.channels}
        chan_scale = {chan: 1.0 for chan in self.channels}

        # update channel property
        time_range = (self.bbox_left, self.bbox_right)
        y0 = - PULSE_STYLE['formatter.margin.top']
        y0_interval = PULSE_STYLE['formatter.margin.between_channel']
        for chan in self.active_channels:
            min_v, max_v = self.chan_event_table[chan].get_min_max(time_range)

            # calculate scaling
            if chan in scales:
                # channel scale factor is specified by user
                scale = scales[chan]
            elif PULSE_STYLE['formatter.control.auto_channel_scaling']:
                # auto scaling is enabled
                max_abs_val = max(abs(max_v), abs(min_v))
                if max_abs_val < PULSE_STYLE['formatter.general.vertical_resolution'] * 100:
                    scale = 1.0
                else:
                    scale = 1 / max_abs_val
            else:
                # not specified by user, no auto scale, then apply default scaling
                if isinstance(chan, pulse.DriveChannel):
                    scale = PULSE_STYLE['formatter.channel_scaling.drive']
                elif isinstance(chan, pulse.ControlChannel):
                    scale = PULSE_STYLE['formatter.channel_scaling.control']
                elif isinstance(chan, pulse.MeasureChannel):
                    scale = PULSE_STYLE['formatter.channel_scaling.measure']
                elif isinstance(chan, pulse.AcquireChannel):
                    scale = PULSE_STYLE['formatter.channel_scaling.acquire']
                else:
                    scale = 1.0

            # keep minimum space
            _min_v = min(PULSE_STYLE['formatter.channel_scaling.min_height'], scale * min_v)
            _max_v = max(PULSE_STYLE['formatter.channel_scaling.max_height'], scale * max_v)

            # calculate offset coordinate
            offset = y0 - _max_v

            # update properties
            chan_visible[chan] = True
            chan_offset[chan] = offset
            chan_scale[chan] = scale

            y0 = offset - (abs(_min_v) + y0_interval)

        # update drawing objects
        for chan in self.channels:
            # update channel info to replace scaling factor
            chan_info = types.ChannelTuple(chan, chan_scale.get(chan, 1.0))
            for gen in PULSE_STYLE['generator.channel']:
                for drawing in gen(chan_info):
                    self._replace_drawing(drawing)

            # update existing drawings
            for drawing in self.drawings:
                if drawing.channel == chan:
                    drawing.visible = chan_visible.get(chan, False)
                    drawing.offset = chan_offset.get(chan, 0.0)
                    drawing.scale = chan_scale.get(chan, 1.0)

        # update boundary box
        self.bbox_bottom = y0 - (PULSE_STYLE['formatter.margin.bottom'] - y0_interval)

        # update axis break
        if PULSE_STYLE['formatter.control.axis_break']:
            self._horizontal_axis_break()

    def _ordered_channels(self,
                          visible_channels: Optional[List[pulse.channels.Channel]] = None) \
            -> List[pulse.channels.Channel]:
        """A helper function to create a list of channels to show.

        Args:
            visible_channels: List of channels to show.
                If not provided, the default channel list is created from the
                stylesheet preference.

        Returns:
            List of ordered channels to show.
        """

        if visible_channels is None:
            channels = []
            for chan in self.channels:
                # remove acquire
                if not PULSE_STYLE['formatter.control.show_acquire_channel'] and \
                        isinstance(chan, pulse.AcquireChannel):
                    continue
                # remove empty
                if not PULSE_STYLE['formatter.control.show_empty_channel'] and \
                        self.chan_event_table[chan].is_empty():
                    continue
                channels.append(chan)
        else:
            channels = visible_channels

        # callback function to arrange channels
        if len(channels) > 1:
            return PULSE_STYLE['layout.channel'](channels)
        else:
            return channels

    def _replace_drawing(self,
                         drawing: drawing_objects.ElementaryData):
        """A helper function to add drawing object.

        If the given drawing object exists in the data container,
        this function just replaces the existing object with the given object
        instead of adding it to the list.

        Args:
            drawing: Drawing object to add to the container.
        """
        if drawing in self.drawings:
            ind = self.drawings.index(drawing)
            self.drawings[ind] = drawing
        else:
            self.drawings.append(drawing)

    def _horizontal_axis_break(self):
        """Generate intervals that are removed from visualization."""
        global_waveform_edges = set()

        for drawing in self.drawings:
            if drawing.data_type in [types.DrawingWaveform.REAL, types.DrawingWaveform.IMAG] \
                    and drawing.visible:
                global_waveform_edges.add(drawing.x[0])
                global_waveform_edges.add(drawing.x[-1])

        global_waveform_edges = sorted(global_waveform_edges)

        event_slacks = []
        for ind in range(1, len(global_waveform_edges)):
            event_slacks.append((global_waveform_edges[ind-1], global_waveform_edges[ind]))

        for event_slack in event_slacks:
            duration = event_slack[1] - event_slack[0]
            if duration > PULSE_STYLE['formatter.axis_break.length']:
                t0 = int(event_slack[0] + 0.5 * PULSE_STYLE['formatter.axis_break.max_length'])
                t1 = int(event_slack[1] - 0.5 * PULSE_STYLE['formatter.axis_break.max_length'])
                self.axis_breaks.append((t0, t1))
