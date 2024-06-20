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

"""
Core module of the pulse drawer.

This module provides the `DrawerCanvas` which is a collection of `Chart` object.
The `Chart` object is a collection of drawings. A user can assign multiple channels
to a single chart instance. For example, we can define a chart for specific qubit
and assign all related channels to the chart. This chart-channel mapping is defined by
the function specified by ``layout.chart_channel_map`` of the stylesheet.

Because this chart instance is decoupled from the coordinate system of the plotter,
we can arbitrarily place charts on the plotter canvas, i.e. if we want to create 3D plot,
each chart may be placed on the X-Z plane and charts are arranged along the Y-axis.
Thus this data model maximizes the flexibility to generate an output image.

The chart instance is not just a container of drawings, as it also performs
data processing like binding abstract coordinates and truncating long pulses for an axis break.
Each chart object has `.parent` which points to the `DrawerCanvas` instance so that
each child chart can refer to the global figure settings such as time range and axis break.


Initialization
~~~~~~~~~~~~~~
The `DataCanvas` and `Chart` are not exposed to users as they are implicitly
initialized in the interface function. It is noteworthy that the data canvas is agnostic
to plotters. This means once the canvas instance is initialized we can reuse this data
among multiple plotters. The canvas is initialized with a stylesheet and quantum backend
information :py:class:`~qiskit.visualization.pulse_v2.device_info.DrawerBackendInfo`.
Chart instances are automatically generated when pulse program is loaded.

    ```python
    canvas = DrawerCanvas(stylesheet=stylesheet, device=device)
    canvas.load_program(sched)
    canvas.update()
    ```

Once all properties are set, `.update` method is called to apply changes to drawings.
If the `DrawDataContainer` is initialized without backend information, the output shows
the time in units of the system cycle time `dt` and the frequencies are initialized to zero.

Update
~~~~~~
To update the image, a user can set new values to canvas and then call the `.update` method.

    ```python
    canvas.set_time_range(2000, 3000, seconds=False)
    canvas.update()
    ```

All stored drawings are updated accordingly. The plotter API can access to
drawings with `.collections` property of chart instance. This returns
an iterator of drawing with the unique data key.
If a plotter provides object handler for plotted shapes, the plotter API can manage
the lookup table of the handler and the drawing by using this data key.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from copy import deepcopy
from enum import Enum
from functools import partial
from itertools import chain

import numpy as np
from qiskit import pulse
from qiskit.pulse.transforms import target_qobj_transform
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import events, types, drawings, device_info
from qiskit.visualization.pulse_v2.stylesheet import QiskitPulseStyle


class DrawerCanvas:
    """Collection of `Chart` and configuration data.

    Pulse channels are associated with some `Chart` instance and
    drawing data object are stored in the `Chart` instance.

    Device, stylesheet, and some user generators are stored in the `DrawingCanvas`
    and `Chart` instances are also attached to the `DrawerCanvas` as children.
    Global configurations are accessed by those children to modify
    the appearance of the `Chart` output.
    """

    def __init__(self, stylesheet: QiskitPulseStyle, device: device_info.DrawerBackendInfo):
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
        self.global_charts = Chart(parent=self, name="global")
        self.charts: list[Chart] = []

        # visible controls
        self.disable_chans: set[pulse.channels.Channel] = set()
        self.disable_types: set[str] = set()

        # data scaling
        self.chan_scales: dict[
            pulse.channels.DriveChannel
            | pulse.channels.MeasureChannel
            | pulse.channels.ControlChannel
            | pulse.channels.AcquireChannel,
            float,
        ] = {}

        # global time
        self._time_range = (0, 0)
        self._time_breaks: list[tuple[int, int]] = []

        # title
        self.fig_title = ""

    @property
    def time_range(self) -> tuple[int, int]:
        """Return current time range to draw.

        Calculate net duration and add side margin to edge location.

        Returns:
            Time window considering side margin.
        """
        t0, t1 = self._time_range

        total_time_elimination = 0
        for t0b, t1b in self.time_breaks:
            if t1b > t0 and t0b < t1:
                total_time_elimination += t1b - t0b
        net_duration = t1 - t0 - total_time_elimination

        new_t0 = t0 - net_duration * self.formatter["margin.left_percent"]
        new_t1 = t1 + net_duration * self.formatter["margin.right_percent"]

        return new_t0, new_t1

    @time_range.setter
    def time_range(self, new_range: tuple[int, int]):
        """Update time range to draw."""
        self._time_range = new_range

    @property
    def time_breaks(self) -> list[tuple[int, int]]:
        """Return time breaks with time range.

        If an edge of time range is in the axis break period,
        the axis break period is recalculated.

        Raises:
            VisualizationError: When axis break is greater than time window.

        Returns:
            List of axis break periods considering the time window edges.
        """
        t0, t1 = self._time_range

        axis_breaks = []
        for t0b, t1b in self._time_breaks:
            if t0b >= t1 or t1b <= t0:
                # skip because break period is outside of time window
                continue

            if t0b < t0 and t1b > t1:
                raise VisualizationError(
                    "Axis break is greater than time window. Nothing will be drawn."
                )
            if t0b < t0 < t1b:
                if t1b - t0 > self.formatter["axis_break.length"]:
                    new_t0 = t0 + 0.5 * self.formatter["axis_break.max_length"]
                    axis_breaks.append((new_t0, t1b))
                continue
            if t0b < t1 < t1b:
                if t1 - t0b > self.formatter["axis_break.length"]:
                    new_t1 = t1 - 0.5 * self.formatter["axis_break.max_length"]
                    axis_breaks.append((t0b, new_t1))
                continue
            axis_breaks.append((t0b, t1b))

        return axis_breaks

    @time_breaks.setter
    def time_breaks(self, new_breaks: list[tuple[int, int]]):
        """Set new time breaks."""
        self._time_breaks = sorted(new_breaks, key=lambda x: x[0])

    def load_program(
        self,
        program: pulse.Waveform | pulse.SymbolicPulse | pulse.Schedule | pulse.ScheduleBlock,
    ):
        """Load a program to draw.

        Args:
            program: Pulse program or waveform to draw.

        Raises:
            VisualizationError: When input program is invalid data format.
        """
        if isinstance(program, (pulse.Schedule, pulse.ScheduleBlock)):
            self._schedule_loader(program)
        elif isinstance(program, (pulse.Waveform, pulse.SymbolicPulse)):
            self._waveform_loader(program)
        else:
            raise VisualizationError(f"Data type {type(program)} is not supported.")

        # update time range
        self.set_time_range(0, program.duration, seconds=False)

        # set title
        self.fig_title = self.layout["figure_title"](program=program, device=self.device)

    def _waveform_loader(
        self,
        program: pulse.Waveform | pulse.SymbolicPulse,
    ):
        """Load Waveform instance.

        This function is sub-routine of py:method:`load_program`.

        Args:
            program: `Waveform` to draw.
        """
        chart = Chart(parent=self)

        # add waveform data
        fake_inst = pulse.Play(program, types.WaveformChannel())
        inst_data = types.PulseInstruction(
            t0=0,
            dt=self.device.dt,
            frame=types.PhaseFreqTuple(phase=0, freq=0),
            inst=fake_inst,
            is_opaque=program.is_parameterized(),
        )
        for gen in self.generator["waveform"]:
            obj_generator = partial(gen, formatter=self.formatter, device=self.device)
            for data in obj_generator(inst_data):
                chart.add_data(data)

        self.charts.append(chart)

    def _schedule_loader(self, program: pulse.Schedule | pulse.ScheduleBlock):
        """Load Schedule instance.

        This function is sub-routine of py:method:`load_program`.

        Args:
            program: `Schedule` to draw.
        """
        program = target_qobj_transform(program, remove_directives=False)

        # initialize scale values
        self.chan_scales = {}
        for chan in program.channels:
            if isinstance(chan, pulse.channels.DriveChannel):
                self.chan_scales[chan] = self.formatter["channel_scaling.drive"]
            elif isinstance(chan, pulse.channels.MeasureChannel):
                self.chan_scales[chan] = self.formatter["channel_scaling.measure"]
            elif isinstance(chan, pulse.channels.ControlChannel):
                self.chan_scales[chan] = self.formatter["channel_scaling.control"]
            elif isinstance(chan, pulse.channels.AcquireChannel):
                self.chan_scales[chan] = self.formatter["channel_scaling.acquire"]
            else:
                self.chan_scales[chan] = 1.0

        # create charts
        mapper = self.layout["chart_channel_map"]
        for name, chans in mapper(
            channels=program.channels, formatter=self.formatter, device=self.device
        ):

            chart = Chart(parent=self, name=name)

            # add standard pulse instructions
            for chan in chans:
                chart.load_program(program=program, chan=chan)

            # add barriers
            barrier_sched = program.filter(
                instruction_types=[pulse.instructions.RelativeBarrier], channels=chans
            )
            for t0, _ in barrier_sched.instructions:
                inst_data = types.BarrierInstruction(t0, self.device.dt, chans)
                for gen in self.generator["barrier"]:
                    obj_generator = partial(gen, formatter=self.formatter, device=self.device)
                    for data in obj_generator(inst_data):
                        chart.add_data(data)

            # add chart axis
            chart_axis = types.ChartAxis(name=chart.name, channels=chart.channels)
            for gen in self.generator["chart"]:
                obj_generator = partial(gen, formatter=self.formatter, device=self.device)
                for data in obj_generator(chart_axis):
                    chart.add_data(data)

            self.charts.append(chart)

        # add snapshot data to global
        snapshot_sched = program.filter(instruction_types=[pulse.instructions.Snapshot])
        for t0, inst in snapshot_sched.instructions:
            inst_data = types.SnapshotInstruction(t0, self.device.dt, inst.label, inst.channels)
            for gen in self.generator["snapshot"]:
                obj_generator = partial(gen, formatter=self.formatter, device=self.device)
                for data in obj_generator(inst_data):
                    self.global_charts.add_data(data)

        # calculate axis break
        self.time_breaks = self._calculate_axis_break(program)

    def _calculate_axis_break(self, program: pulse.Schedule) -> list[tuple[int, int]]:
        """A helper function to calculate axis break of long pulse sequence.

        Args:
            program: A schedule to calculate axis break.

        Returns:
            List of axis break periods.
        """
        axis_breaks = []

        edges = set()
        for t0, t1 in chain.from_iterable(program.timeslots.values()):
            if t1 - t0 > 0:
                edges.add(t0)
                edges.add(t1)
        edges = sorted(edges)

        for t0, t1 in zip(edges[:-1], edges[1:]):
            if t1 - t0 > self.formatter["axis_break.length"]:
                t_l = t0 + 0.5 * self.formatter["axis_break.max_length"]
                t_r = t1 - 0.5 * self.formatter["axis_break.max_length"]
                axis_breaks.append((t_l, t_r))

        return axis_breaks

    def set_time_range(self, t_start: float, t_end: float, seconds: bool = True):
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
                raise VisualizationError(
                    "Setting time range with SI units requires backend `dt` information."
                )
        self.time_range = (t_start, t_end)

    def set_disable_channel(self, channel: pulse.channels.Channel, remove: bool = True):
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

    def set_disable_type(self, data_type: types.DataTypes, remove: bool = True):
        """Interface method to control visibility of data types.

        Specified object in the blocked list will not be shown.

        Args:
            data_type: A drawing data type to disable.
            remove: Set `True` to disable, set `False` to enable.
        """
        if isinstance(data_type, Enum):
            data_type_str = str(data_type.value)
        else:
            data_type_str = data_type

        if remove:
            self.disable_types.add(data_type_str)
        else:
            self.disable_types.discard(data_type_str)

    def update(self):
        """Update all associated charts and generate actual drawing data from template object.

        This method should be called before the canvas is passed to the plotter.
        """
        for chart in self.charts:
            chart.update()


class Chart:
    """A collection of drawing to be shown on the same line.

    Multiple pulse channels can be assigned to a single `Chart`.
    The parent `DrawerCanvas` should be specified to refer to the current user preference.

    The vertical value of each `Chart` should be in the range [-1, 1].
    This truncation should be performed in the plotter interface.
    """

    # unique index of chart
    chart_index = 0

    # list of waveform type names
    waveform_types = [
        str(types.WaveformType.REAL.value),
        str(types.WaveformType.IMAG.value),
        str(types.WaveformType.OPAQUE.value),
    ]

    def __init__(self, parent: DrawerCanvas, name: str | None = None):
        """Create new chart.

        Args:
            parent: `DrawerCanvas` that this `Chart` instance belongs to.
            name: Name of this `Chart` instance.
        """
        self.parent = parent

        # data stored in this channel
        self._collections: dict[str, drawings.ElementaryData] = {}
        self._output_dataset: dict[str, drawings.ElementaryData] = {}

        # channel metadata
        self.index = self._cls_index()
        self.name = name or ""
        self._channels: set[pulse.channels.Channel] = set()

        # vertical axis information
        self.vmax = 0
        self.vmin = 0
        self.scale = 1.0

        self._increment_cls_index()

    def add_data(self, data: drawings.ElementaryData):
        """Add drawing to collections.

        If the given object already exists in the collections,
        this interface replaces the old object instead of adding new entry.

        Args:
            data: New drawing to add.
        """
        self._collections[data.data_key] = data

    def load_program(self, program: pulse.Schedule, chan: pulse.channels.Channel):
        """Load pulse schedule.

        This method internally generates `ChannelEvents` to parse the program
        for the specified pulse channel. This method is called once

        Args:
            program: Pulse schedule to load.
            chan: A pulse channels associated with this instance.
        """
        chan_events = events.ChannelEvents.load_program(program, chan)
        chan_events.set_config(
            dt=self.parent.device.dt,
            init_frequency=self.parent.device.get_channel_frequency(chan),
            init_phase=0,
        )

        # create objects associated with waveform
        for gen in self.parent.generator["waveform"]:
            waveforms = chan_events.get_waveforms()
            obj_generator = partial(gen, formatter=self.parent.formatter, device=self.parent.device)
            drawing_items = [obj_generator(waveform) for waveform in waveforms]
            for drawing_item in list(chain.from_iterable(drawing_items)):
                self.add_data(drawing_item)

        # create objects associated with frame change
        for gen in self.parent.generator["frame"]:
            frames = chan_events.get_frame_changes()
            obj_generator = partial(gen, formatter=self.parent.formatter, device=self.parent.device)
            drawing_items = [obj_generator(frame) for frame in frames]
            for drawing_item in list(chain.from_iterable(drawing_items)):
                self.add_data(drawing_item)

        self._channels.add(chan)

    def update(self):
        """Update vertical data range and scaling factor of this chart.

        Those parameters are updated based on current time range in the parent canvas.
        """
        self._output_dataset.clear()
        self.vmax = 0
        self.vmin = 0

        # waveform
        for key, data in self._collections.items():
            if data.data_type not in Chart.waveform_types:
                continue

            # truncate, assume no abstract coordinate in waveform sample
            trunc_x, trunc_y = self._truncate_data(data)

            # no available data points
            if trunc_x.size == 0 or trunc_y.size == 0:
                continue

            # update y range
            scale = min(self.parent.chan_scales.get(chan, 1.0) for chan in data.channels)
            self.vmax = max(scale * np.max(trunc_y), self.vmax)
            self.vmin = min(scale * np.min(trunc_y), self.vmin)

            # generate new data
            new_data = deepcopy(data)
            new_data.xvals = trunc_x
            new_data.yvals = trunc_y

            self._output_dataset[key] = new_data

        # calculate chart level scaling factor
        if self.parent.formatter["control.auto_chart_scaling"]:
            max_val = max(
                abs(self.vmax), abs(self.vmin), self.parent.formatter["general.vertical_resolution"]
            )
            self.scale = min(1.0 / max_val, self.parent.formatter["general.max_scale"])
        else:
            self.scale = 1.0

        # update vertical range with scaling and limitation
        self.vmax = max(
            self.scale * self.vmax, self.parent.formatter["channel_scaling.pos_spacing"]
        )

        self.vmin = min(
            self.scale * self.vmin, self.parent.formatter["channel_scaling.neg_spacing"]
        )

        # other data
        for key, data in self._collections.items():
            if data.data_type in Chart.waveform_types:
                continue

            # truncate
            trunc_x, trunc_y = self._truncate_data(data)

            # no available data points
            if trunc_x.size == 0 or trunc_y.size == 0:
                continue

            # generate new data
            new_data = deepcopy(data)
            new_data.xvals = trunc_x
            new_data.yvals = trunc_y

            self._output_dataset[key] = new_data

    @property
    def is_active(self) -> bool:
        """Check if there is any active waveform data in this entry.

        Returns:
            Return `True` if there is any visible waveform in this chart.
        """
        for data in self._output_dataset.values():
            if data.data_type in Chart.waveform_types and self._check_visible(data):
                return True
        return False

    @property
    def collections(self) -> Iterator[tuple[str, drawings.ElementaryData]]:
        """Return currently active entries from drawing data collection.

        The object is returned with unique name as a key of an object handler.
        When the horizontal coordinate contains `AbstractCoordinate`,
        the value is substituted by current time range preference.
        """
        for name, data in self._output_dataset.items():
            # prepare unique name
            unique_id = f"chart{self.index:d}_{name}"
            if self._check_visible(data):
                yield unique_id, data

    @property
    def channels(self) -> list[pulse.channels.Channel]:
        """Return a list of channels associated with this chart.

        Returns:
            List of channels associated with this chart.
        """
        return list(self._channels)

    def _truncate_data(self, data: drawings.ElementaryData) -> tuple[np.ndarray, np.ndarray]:
        """A helper function to truncate drawings according to time breaks.

        # TODO: move this function to common module to support axis break for timeline.

        Args:
            data: Drawing object to truncate.

        Returns:
            Set of truncated numpy arrays for x and y coordinate.
        """
        xvals = self._bind_coordinate(data.xvals)
        yvals = self._bind_coordinate(data.yvals)

        if isinstance(data, drawings.BoxData):
            # truncate box data. these object don't require interpolation at axis break.
            return self._truncate_boxes(xvals, yvals)
        elif data.data_type in [types.LabelType.PULSE_NAME, types.LabelType.OPAQUE_BOXTEXT]:
            # truncate pulse labels. these objects are not removed by truncation.
            return self._truncate_pulse_labels(xvals, yvals)
        else:
            # other objects
            return self._truncate_vectors(xvals, yvals)

    def _truncate_pulse_labels(
        self, xvals: np.ndarray, yvals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """A helper function to remove text according to time breaks.

        Args:
            xvals: Time points.
            yvals: Data points.

        Returns:
            Set of truncated numpy arrays for x and y coordinate.
        """
        xpos = xvals[0]
        t0, t1 = self.parent.time_range

        if xpos < t0 or xpos > t1:
            return np.array([]), np.array([])
        offset_accumulation = 0
        for tl, tr in self.parent.time_breaks:
            if xpos < tl:
                return np.array([xpos - offset_accumulation]), yvals
            if tl < xpos < tr:
                return np.array([tl - offset_accumulation]), yvals
            else:
                offset_accumulation += tr - tl
        return np.array([xpos - offset_accumulation]), yvals

    def _truncate_boxes(
        self, xvals: np.ndarray, yvals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """A helper function to clip box object according to time breaks.

        Args:
            xvals: Time points.
            yvals: Data points.

        Returns:
            Set of truncated numpy arrays for x and y coordinate.
        """
        x0, x1 = xvals
        t0, t1 = self.parent.time_range

        if x1 < t0 or x0 > t1:
            # out of drawing range
            return np.array([]), np.array([])

        # clip outside
        x0 = max(t0, x0)
        x1 = min(t1, x1)

        offset_accumulate = 0
        for tl, tr in self.parent.time_breaks:
            tl -= offset_accumulate
            tr -= offset_accumulate

            #
            # truncate, there are 5 patterns wrt the relative position of truncation and xvals
            #
            if x1 < tl:
                break

            if tl < x0 and tr > x1:
                # case 1: all data points are truncated
                #      :   +-----+   :
                #      :   |/////|   :
                # -----:---+-----+---:-----
                #      l   0     1   r
                return np.array([]), np.array([])
            elif tl < x1 < tr:
                # case 2: t < tl, right side is truncated
                #      +---:-----+   :
                #      |   ://///|   :
                # -----+---:-----+---:-----
                #      0   l     1   r
                x1 = tl
            elif tl < x0 < tr:
                # case 3: tr > t, left side is truncated
                #      :   +-----:---+
                #      :   |/////:   |
                # -----:---+-----:---+-----
                #      l   0     r   1
                x0 = tl
                x1 = tl + t1 - tr
            elif tl > x0 and tr < x1:
                # case 4: tr > t > tl, middle part is truncated
                #      +---:-----:---+
                #      |   ://///:   |
                # -----+---:-----:---+-----
                #      0   l     r   1
                x1 -= tr - tl
            elif tr < x0:
                # case 5: tr > t > tl, nothing truncated but need time shift
                #      :   :     +---+
                #      :   :     |   |
                # -----:---:-----+---+-----
                #      l   r     0   1
                x0 -= tr - tl
                x1 -= tr - tl

            offset_accumulate += tr - tl

        return np.asarray([x0, x1], dtype=float), yvals

    def _truncate_vectors(
        self, xvals: np.ndarray, yvals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """A helper function to remove sequential data points according to time breaks.

        Args:
            xvals: Time points.
            yvals: Data points.

        Returns:
            Set of truncated numpy arrays for x and y coordinate.
        """
        xvals = np.asarray(xvals, dtype=float)
        yvals = np.asarray(yvals, dtype=float)
        t0, t1 = self.parent.time_range

        if max(xvals) < t0 or min(xvals) > t1:
            # out of drawing range
            return np.array([]), np.array([])

        if min(xvals) < t0:
            # truncate x less than left limit
            inds = xvals > t0
            yvals = np.append(np.interp(t0, xvals, yvals), yvals[inds])
            xvals = np.append(t0, xvals[inds])

        if max(xvals) > t1:
            # truncate x larger than right limit
            inds = xvals < t1
            yvals = np.append(yvals[inds], np.interp(t1, xvals, yvals))
            xvals = np.append(xvals[inds], t1)

        # time breaks
        trunc_xvals = [xvals]
        trunc_yvals = [yvals]
        offset_accumulate = 0
        for tl, tr in self.parent.time_breaks:
            sub_xs = trunc_xvals.pop()
            sub_ys = trunc_yvals.pop()
            tl -= offset_accumulate
            tr -= offset_accumulate

            #
            # truncate, there are 5 patterns wrt the relative position of truncation and xvals
            #
            min_xs = min(sub_xs)
            max_xs = max(sub_xs)
            if max_xs < tl:
                trunc_xvals.append(sub_xs)
                trunc_yvals.append(sub_ys)
                break

            if tl < min_xs and tr > max_xs:
                # case 1: all data points are truncated
                #      :   +-----+   :
                #      :   |/////|   :
                # -----:---+-----+---:-----
                #      l  min   max  r
                return np.array([]), np.array([])
            elif tl < max_xs < tr:
                # case 2: t < tl, right side is truncated
                #      +---:-----+   :
                #      |   ://///|   :
                # -----+---:-----+---:-----
                #     min  l    max  r
                inds = sub_xs > tl
                trunc_xvals.append(np.append(tl, sub_xs[inds]) - (tl - min_xs))
                trunc_yvals.append(np.append(np.interp(tl, sub_xs, sub_ys), sub_ys[inds]))
            elif tl < min_xs < tr:
                # case 3: tr > t, left side is truncated
                #      :   +-----:---+
                #      :   |/////:   |
                # -----:---+-----:---+-----
                #      l  min    r  max
                inds = sub_xs < tr
                trunc_xvals.append(np.append(sub_xs[inds], tr))
                trunc_yvals.append(np.append(sub_ys[inds], np.interp(tr, sub_xs, sub_ys)))
            elif tl > min_xs and tr < max_xs:
                # case 4: tr > t > tl, middle part is truncated
                #      +---:-----:---+
                #      |   ://///:   |
                # -----+---:-----:---+-----
                #     min  l     r  max
                inds0 = sub_xs < tl
                trunc_xvals.append(np.append(sub_xs[inds0], tl))
                trunc_yvals.append(np.append(sub_ys[inds0], np.interp(tl, sub_xs, sub_ys)))
                inds1 = sub_xs > tr
                trunc_xvals.append(np.append(tr, sub_xs[inds1]) - (tr - tl))
                trunc_yvals.append(np.append(np.interp(tr, sub_xs, sub_ys), sub_ys[inds1]))
            elif tr < min_xs:
                # case 5: tr > t > tl, nothing truncated but need time shift
                #      :   :     +---+
                #      :   :     |   |
                # -----:---:-----+---+-----
                #      l   r     0   1
                trunc_xvals.append(sub_xs - (tr - tl))
                trunc_yvals.append(sub_ys)
            else:
                # no need to truncate
                trunc_xvals.append(sub_xs)
                trunc_yvals.append(sub_ys)
            offset_accumulate += tr - tl

        new_x = np.concatenate(trunc_xvals)
        new_y = np.concatenate(trunc_yvals)

        return np.asarray(new_x, dtype=float), np.asarray(new_y, dtype=float)

    def _bind_coordinate(self, vals: Sequence[types.Coordinate] | np.ndarray) -> np.ndarray:
        """A helper function to bind actual coordinates to an `AbstractCoordinate`.

        Args:
            vals: Sequence of coordinate objects associated with a drawing.

        Returns:
            Numpy data array with substituted values.
        """

        def substitute(val: types.Coordinate):
            if val == types.AbstractCoordinate.LEFT:
                return self.parent.time_range[0]
            if val == types.AbstractCoordinate.RIGHT:
                return self.parent.time_range[1]
            if val == types.AbstractCoordinate.TOP:
                return self.vmax
            if val == types.AbstractCoordinate.BOTTOM:
                return self.vmin
            raise VisualizationError(f"Coordinate {val} is not supported.")

        try:
            return np.asarray(vals, dtype=float)
        except (TypeError, ValueError):
            return np.asarray(list(map(substitute, vals)), dtype=float)

    def _check_visible(self, data: drawings.ElementaryData) -> bool:
        """A helper function to check if the data is visible.

        Args:
            data: Drawing object to test.

        Returns:
            Return `True` if the data is visible.
        """
        is_active_type = data.data_type not in self.parent.disable_types
        is_active_chan = any(chan not in self.parent.disable_chans for chan in data.channels)
        if not (is_active_type and is_active_chan):
            return False

        return True

    @classmethod
    def _increment_cls_index(cls):
        """Increment counter of the chart."""
        cls.chart_index += 1

    @classmethod
    def _cls_index(cls) -> int:
        """Return counter index of the chart."""
        return cls.chart_index
