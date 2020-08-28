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

from copy import deepcopy
from functools import partial
from itertools import chain
from typing import Union, List, Tuple, Iterator

import numpy as np

from qiskit import pulse
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import events, types, drawing_objects, device_info
from qiskit.visualization.pulse_v2.style.stylesheet import QiskitPulseStyle


class DrawerCanvas:
    """Collection of `Chart` and configuration data.

    Pulse channels are associated with some `Chart` instance and
    drawing data object is stored in the `Chart` instance.

    Device, stylesheet, and some user preferences are stored in the `DrawingCanvas`
    and the `Chart` instances are also attached to the `DrawerCanvas` as children.
    Global configurations are accessed by those children to modify appearance of `Chart` output.
    """

    def __init__(self,
                 stylesheet: QiskitPulseStyle,
                 device: device_info.DrawerBackendInfo):
        """Create new data container with backend system information.

        Args:
            stylesheet: Stylesheet to decide appearance of output image.
            device: Backend information to run the program.
        """

        # stylesheet
        self.formatter = stylesheet.formatter
        self.generator = stylesheet.generator
        self.layout = stylesheet.layout

        # device info
        self.device = device

        # chart
        self.charts = []

        # visible controls
        self.disable_chans = set()
        self.disable_types = set()

        # data scaling
        self.chan_scales = dict()

        # global time
        self._time_range = (0, 0)
        self._time_breaks = []

    @property
    def time_range(self):
        """Return current time range to draw."""
        return self._time_range

    @time_range.setter
    def time_range(self, new_range: Tuple[int, int]):
        """Update time range to draw and update child charts."""
        self._time_range = new_range
        for chart in self.charts:
            chart.update()

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

        # update time range
        self.set_time_range(0, program.duration)

    def _waveform_loader(self, program: Union[pulse.Waveform, pulse.ParametricPulse]):
        """Load Waveform instance.

        This function is sub-routine of py:method:`load_program`.

        Args:
            program: `Waveform` to draw.
        """
        chart_name = ''

        chart = Chart(parent=self, name=chart_name)

        # add waveform data
        sample_channel = types.WaveformChannel()
        inst_data = types.InstructionTuple(t0=0,
                                           dt=self.device.dt,
                                           frame=types.PhaseFreqTuple(phase=0, freq=0),
                                           inst=pulse.Play(program, sample_channel))
        for gen in self.generator['waveform']:
            obj_generator = partial(func=gen,
                                    formatter=self.formatter,
                                    device=self.device)
            for data in obj_generator(inst_data):
                chart.add_data(data)

        # add chart axis
        chart_header = types.ChartAxis(name=chart_name)
        for gen in self.generator['chart']:
            obj_generator = partial(func=gen,
                                    formatter=self.formatter,
                                    device=self.device)
            for data in obj_generator(chart_header):
                chart.add_data(data)

        self.charts.append(chart)

    def _schedule_loader(self, program: pulse.Schedule):
        """Load Schedule instance.

        This function is sub-routine of py:method:`load_program`.

        Args:
            program: `Schedule` to draw.
        """
        # initialize scale values
        self.chan_scales = {chan: 1.0 for chan in program.channels}

        # create charts
        mapper = self.layout['chart_channel_map']
        for name, chans in mapper(channels=program.channels,
                                  formatter=self.formatter,
                                  device=self.device):
            chart = Chart(parent=self, name=name)

            # add standard pulse instructions
            for chan in chans:
                chart.load_program(program=program, chan=chan)

            # add barriers
            barrier_sched = program.filter(instruction_types=[pulse.instructions.RelativeBarrier],
                                           channels=chans)
            for t0, _ in barrier_sched.instructions:
                inst_data = types.Barrier(t0, self.device.dt, chans)
                for gen in self.generator['barrier']:
                    obj_generator = partial(func=gen,
                                            formatter=self.formatter,
                                            device=self.device)
                    for data in obj_generator(inst_data):
                        chart.add_data(data)

            # add chart axis
            chart_header = types.ChartAxis(name=name)
            for gen in self.generator['chart']:
                obj_generator = partial(func=gen,
                                        formatter=self.formatter,
                                        device=self.device)
                for data in obj_generator(chart_header):
                    chart.add_data(data)

            self.charts.append(chart)

        # create snapshot chart
        snapshot_chart = Chart(parent=self, name='snapshot')

        snapshot_sched = program.filter(instruction_types=[pulse.instructions.Snapshot])
        for t0, inst in snapshot_sched.instructions:
            inst_data = types.Snapshots(t0, self.device.dt, inst.channels)
            for gen in self.generator['snapshot']:
                obj_generator = partial(func=gen,
                                        formatter=self.formatter,
                                        device=self.device)
                for data in obj_generator(inst_data):
                    snapshot_chart.add_data(data)
        self.charts.append(snapshot_chart)

        # calculate axis break
        self._time_breaks = self._calculate_axis_break(program)

    def _calculate_axis_break(self, program: pulse.Schedule) -> List[Tuple[int, int]]:
        """A helper function to calculate axis break of long pulse sequence.

        Args:
            program: A schedule to calculate axis break.
        """
        axis_breaks = []

        edges = set()
        for t0, t1 in chain.from_iterable(program.timeslots.values()):
            if t1 - t0 > 0:
                edges.add(t0)
                edges.add(t1)
        edges = sorted(edges)

        for t0, t1 in zip(edges[:-1], edges[1:]):
            if t1 - t0 > self.formatter['axis_break.length']:
                t_l = t0 + 0.5 * self.formatter['axis_break.max_length']
                t_r = t1 - 0.5 * self.formatter['axis_break.max_length']
                axis_breaks.append((t_l, t_r))

        return axis_breaks

    def set_time_range(self,
                       t_start: Union[int, float],
                       t_end: Union[int, float],
                       seconds: bool = True):
        """Set time range to draw.

        All child chart instances are updated when time range is updated.

        Args:
            t_start: Left boundary of drawing in units of cycle time or real time.
            t_end: Right boundary of drawing in units of cycle time or real time.
            seconds: Set `True` if times are given in SI unit rather than dt.

        Raises:
            VisualizationError: When times are given in float without specifying dt.
        """
        # convert into nearest cycle time
        if seconds:
            if self.device.dt is not None:
                t_start = int(np.round(t_start / self.device.dt))
                t_end = int(np.round(t_end / self.device.dt))
            else:
                raise VisualizationError('Setting time range with SI units requires '
                                         'backend `dt` information.')
        self.time_range = (t_start, t_end)

    def set_disable_channel(self,
                            channel: pulse.channels.Channel,
                            remove: bool = True):
        """Interface method to control visibility of pulse channels.

        Specified object in the blocked list will not be shown.

        Args:
            channel: A pulse channel object to disable.
            remove: Set `True` to disable, set `False` to enable.
        """
        if remove:
            self.disable_chans.add(channel)
        else:
            self.disable_chans.discard(channel)

    def set_disable_type(self,
                         data_type: types.DataTypes,
                         remove: bool = True):
        """Interface method to control visibility of data types.

        Specified object in the blocked list will not be shown.

        Args:
            data_type: A drawing object data type to disable.
            remove: Set `True` to disable, set `False` to enable.
        """
        if remove:
            self.disable_types.add(data_type)
        else:
            self.disable_types.discard(data_type)


class Chart:
    """A collection of drawing object to be shown in the same line.

    Multiple pulse channels can be assigned to a single `Chart`.
    The parent `DrawerCanvas` should be specified to refer to the current user preference.

    The vertical value of each `Chart` should be in the range [-1, 1].
    This truncation should be performed in the plotter interface.
    """
    WAVEFORMS = (types.DrawingWaveform.REAL, types.DrawingWaveform.IMAG)

    # unique index of chart
    chart_index = 0

    def __init__(self, parent: DrawerCanvas, name: str):
        """Create new chart.

        Args:
            parent: `DrawerCanvas` that this `Chart` instance belongs to.
            name: Name of this `Chart` instance.
        """
        self._parent = parent
        self._collections = []

        self.index = self._cls_index()
        self.name = name

        self.vmax = 0
        self.vmin = 0
        self.scale = 1.0

        self._increment_cls_index()

    def add_data(self, data: drawing_objects.ElementaryData):
        """Add drawing object to collections.

        If the given object already exists in the collections,
        this interface replaces the old object instead of adding new entry.

        Args:
            data: New drawing object to add.
        """
        if data in self._collections:
            ind = self._collections.index(data)
            self._collections[ind] = data
        else:
            self._collections.append(data)

    def load_program(self,
                     program: pulse.Schedule,
                     chan: pulse.channels.Channel):
        """Load pulse schedule.

        This method internally generates `ChannelEvents` to parse the program
        for the specified pulse channel. This method is called once

        Args:
            program: Pulse schedule to load.
            chan: A pulse channels associated with this instance.
        """
        chan_events = events.ChannelEvents.load_program(program, chan)
        chan_events.config(dt=self._parent.device.dt,
                           init_frequency=self._parent.device.get_channel_frequency(chan),
                           init_phase=0)

        # create objects associated with waveform
        waveforms = chan_events.get_waveforms()
        for gen in self._parent.generator['waveform']:
            obj_generator = partial(func=gen,
                                    formatter=self._parent.formatter,
                                    device=self._parent.device)
            drawings = [obj_generator(waveform) for waveform in waveforms]
            for data in list(chain.from_iterable(drawings)):
                self.add_data(data)

        # create objects associated with frame change
        frames = chan_events.get_frame_changes()
        for gen in self._parent.generator['frame']:
            obj_generator = partial(func=gen,
                                    formatter=self._parent.formatter,
                                    device=self._parent.device)
            drawings = [obj_generator(frame) for frame in frames]
            for data in list(chain.from_iterable(drawings)):
                self.add_data(data)

    def update(self):
        """Update vertical data range and scaling factor of this chart.

        Those parameters are updated based on current time range in the parent canvas.
        """
        for _, data in self.collections:
            if data.data_type in Chart.WAVEFORMS:
                scale = min(self._parent.chan_scales.get(chan, 1.0) for chan in data.channels)
                self.vmax = max(scale * data.vmax, self.vmax)
                self.vmin = max(scale * data.vmin, self.vmin)

        if self._parent.formatter['control.auto_chart_scaling']:
            max_scaling = self._parent.formatter['general.vertical_resolution']
            self.scale = 1.0 / (max(abs(self.vmax), abs(self.vmin), max_scaling))
        else:
            self.scale = 1.0

    @property
    def is_active(self):
        """Check if there is any active waveform data in this entry."""
        for _, data in self.collections:
            if data.data_type in Chart.WAVEFORMS:
                return True
        return False

    @property
    def collections(self) -> Iterator[Tuple[str, drawing_objects.ElementaryData]]:
        """Return currently active entries from drawing data collection.

        The object is returned with unique name as a key of an object handler.
        When the horizontal coordinate contains `AbstractCoordinate`,
        the value is substituted by current time range preference.
        """
        t0, t1 = self._parent.time_range

        for data in self._collections:
            # prepare unique name
            data_name = 'chart{ind:d}_{key}'.format(ind=self.index, key=data.data_key)

            is_active_type = data.data_type in self._parent.disable_types
            is_active_chan = any(chan not in self._parent.disable_chans for chan in data.channels)

            # skip the entry if channel or data type are in the blocked list.
            if not (is_active_type and is_active_chan):
                continue

            # skip the entry if location is out of time range.
            try:
                if isinstance(data.x, types.Coordinate):
                    # single entry
                    x_arr = self._bind_coordinate([data.x])
                    new_x = x_arr[0]
                else:
                    # iterator
                    x_arr = self._bind_coordinate(data.x)
                    new_x = x_arr

                if not any(np.where((x_arr >= t0) & (x_arr <= t1), True, False)):
                    continue

                # prepare new entry which doesn't contain abstract coordinate.
                # this substitution changes the data key, thus object is deep copied
                # to avoid duplication of the same entry in the collection.
                new_data = deepcopy(data)
                new_data.x = new_x

                yield data_name, new_data

            except AttributeError:
                # data has no x attribute. likely no dependence on time range.
                yield data_name, data

    def _bind_coordinate(self, xvals: Iterator[types.Coordinate]) -> np.ndarray:
        """A helper function to bind actual coordinate to `AbstractCoordinate`.

        Args:
            xvals: Arbitrary coordinate object associated with a drawing object.
        """
        def substitute(xval: types.Coordinate):
            if xval == types.AbstractCoordinate.LEFT:
                return self._parent.time_range[0]
            if xval == types.AbstractCoordinate.RIGHT:
                return self._parent.time_range[1]
            return xval

        try:
            return np.asarray(xvals, dtype=float)
        except ValueError:
            return np.asarray(list(map(substitute, xvals)), dtype=float)

    @classmethod
    def _increment_cls_index(cls):
        """Increment counter of the chart."""
        cls.chart_index += 1

    @classmethod
    def _cls_index(cls):
        """Return counter index of the chart."""
        return cls.chart_index
