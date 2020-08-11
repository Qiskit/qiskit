# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Matplotlib classes for pulse visualization."""

import collections
from typing import Dict, List, Tuple, Callable, Union, Any

import numpy as np

try:
    from matplotlib import pyplot as plt, gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from qiskit.visualization.pulse.qcstyle import PulseStyle, SchedStyle
from qiskit.visualization.pulse.interpolation import step_wise
from qiskit.pulse.channels import (DriveChannel, ControlChannel,
                                   MeasureChannel, AcquireChannel,
                                   SnapshotChannel, Channel)
from qiskit.pulse import (Waveform, SamplePulse, Snapshot, Play,
                          Acquire, PulseError, ParametricPulse, SetFrequency, ShiftPhase,
                          Instruction, ScheduleComponent, ShiftFrequency, SetPhase)


class EventsOutputChannels:
    """Pulse dataset for channel."""

    def __init__(self, t0: int, tf: int):
        """Create new channel dataset.

        TODO: remove PV

        Args:
            t0: starting time of plot
            tf: ending time of plot
        """
        self.pulses = {}
        self.t0 = t0
        self.tf = tf

        self._waveform = None
        self._framechanges = None
        self._setphase = None
        self._frequencychanges = None
        self._conditionals = None
        self._snapshots = None
        self._labels = None
        self.enable = False

    def add_instruction(self, start_time: int, instruction: Instruction):
        """Add new pulse instruction to channel.

        Args:
            start_time: Starting time of instruction
            instruction: Instruction object to be added
        """
        if isinstance(instruction, Play):
            pulse = instruction.pulse
        else:
            pulse = instruction
        if start_time in self.pulses.keys():
            self.pulses[start_time].append(pulse)
        else:
            self.pulses[start_time] = [pulse]

    @property
    def waveform(self) -> np.ndarray:
        """Get waveform."""
        if self._waveform is None:
            self._build_waveform()

        return self._waveform[self.t0:self.tf]

    @property
    def framechanges(self) -> Dict[int, ShiftPhase]:
        """Get frame changes."""
        if self._framechanges is None:
            self._build_waveform()

        return self._trim(self._framechanges)

    @property
    def setphase(self) -> Dict[int, SetPhase]:
        """Get the SetPhase phase values."""
        if self._setphase is None:
            self._build_waveform()

        return self._trim(self._setphase)

    @property
    def frequencychanges(self) -> Dict[int, SetFrequency]:
        """Get the frequency changes."""
        if self._frequencychanges is None:
            self._build_waveform()

        return self._trim(self._frequencychanges)

    @property
    def frequencyshift(self) -> Dict[int, ShiftFrequency]:
        """Set the frequency changes."""
        if self._frequencychanges is None:
            self._build_waveform()

        return self._trim(self._frequencychanges)

    @property
    def conditionals(self) -> Dict[int, str]:
        """Get conditionals."""
        if self._conditionals is None:
            self._build_waveform()

        return self._trim(self._conditionals)

    @property
    def snapshots(self) -> Dict[int, Snapshot]:
        """Get snapshots."""
        if self._snapshots is None:
            self._build_waveform()

        return self._trim(self._snapshots)

    @property
    def labels(self) -> Dict[int, Union[Waveform, Acquire]]:
        """Get labels."""
        if self._labels is None:
            self._build_waveform()

        return self._trim(self._labels)

    def is_empty(self) -> bool:
        """Return if pulse is empty.

        Returns:
            bool: if the channel has nothing to plot
        """
        if (any(self.waveform) or self.framechanges or self.setphase or
                self.conditionals or self.snapshots):
            return False

        return True

    def to_table(self, name: str) -> List[Tuple[int, str, str]]:
        """Get table contains.

        Args:
            name (str): name of channel

        Returns:
            A list of events in the channel
        """
        time_event = []

        framechanges = self.framechanges
        setphase = self.setphase
        conditionals = self.conditionals
        snapshots = self.snapshots
        frequencychanges = self.frequencychanges

        for key, val in framechanges.items():
            data_str = 'shift phase: %.2f' % val
            time_event.append((key, name, data_str))
        for key, val in setphase.items():
            data_str = 'set phase: %.2f' % val
            time_event.append((key, name, data_str))
        for key, val in conditionals.items():
            data_str = 'conditional, %s' % val
            time_event.append((key, name, data_str))
        for key, val in snapshots.items():
            data_str = 'snapshot: %s' % val
            time_event.append((key, name, data_str))
        for key, val in frequencychanges.items():
            data_str = 'frequency: %.4e' % val
            time_event.append((key, name, data_str))

        return time_event

    def _build_waveform(self):
        """Create waveform from stored pulses.
        """
        self._framechanges = {}
        self._setphase = {}
        self._frequencychanges = {}
        self._conditionals = {}
        self._snapshots = {}
        self._labels = {}
        fc = 0
        pv = np.zeros(self.tf + 1, dtype=np.complex128)
        wf = np.zeros(self.tf + 1, dtype=np.complex128)
        for time, commands in sorted(self.pulses.items()):
            if time > self.tf:
                break
            tmp_fc = 0
            tmp_set_phase = 0
            tmp_sf = None
            for command in commands:
                if isinstance(command, ShiftPhase):
                    tmp_fc += command.phase
                    pv[time:] = 0
                elif isinstance(command, SetPhase):
                    tmp_set_phase = command.phase
                    pv[time:] = 0
                elif isinstance(command, SetFrequency):
                    tmp_sf = command.frequency
                elif isinstance(command, ShiftFrequency):
                    tmp_sf = command.frequency
                elif isinstance(command, Snapshot):
                    self._snapshots[time] = command.name
            if tmp_fc != 0:
                self._framechanges[time] = tmp_fc
                fc += tmp_fc
            if tmp_set_phase != 0:
                self._setphase[time] = tmp_set_phase
                fc = tmp_set_phase
            if tmp_sf is not None:
                self._frequencychanges[time] = tmp_sf

            for command in commands:
                duration = command.duration
                tf = min(time + duration, self.tf)
                if isinstance(command, ParametricPulse):
                    command = command.get_waveform()
                if isinstance(command, (Waveform, SamplePulse)):
                    wf[time:tf] = np.exp(1j*fc) * command.samples[:tf-time]
                    pv[time:] = 0
                    self._labels[time] = (tf, command)

                elif isinstance(command, Acquire):
                    wf[time:tf] = np.ones(tf - time)
                    self._labels[time] = (tf, command)
        self._waveform = wf + pv

    def _trim(self, events: Dict[int, Any]) -> Dict[int, Any]:
        """Return events during given `time_range`.

        Args:
            events: time and operation of events.

        Returns:
            Events within the specified time range.
        """
        events_in_time_range = {}

        for k, v in events.items():
            if self.t0 <= k <= self.tf:
                events_in_time_range[k] = v

        return events_in_time_range


class WaveformDrawer:
    """A class to create figure for sample pulse."""

    def __init__(self, style: PulseStyle):
        """Create new figure.

        Args:
            style: Style sheet for pulse visualization.
        """
        self.style = style or PulseStyle()

    def draw(self, pulse: Waveform,
             dt: float = 1.0,
             interp_method: Callable = None,
             scale: float = 1):
        """Draw figure.

        Args:
            pulse: Waveform to draw.
            dt: time interval.
            interp_method: interpolation function.
            scale: Relative visual scaling of waveform amplitudes.

        Returns:
            matplotlib.figure.Figure: A matplotlib figure object of the pulse envelope.
        """
        # If these self.style.dpi or self.style.figsize are None, they will
        # revert back to their default rcParam keys.
        figure = plt.figure(dpi=self.style.dpi, figsize=self.style.figsize)

        interp_method = interp_method or step_wise

        ax = figure.add_subplot(111)
        ax.set_facecolor(self.style.bg_color)

        samples = pulse.samples
        time = np.arange(0, len(samples) + 1, dtype=float) * dt

        time, re, im = interp_method(time, samples, self.style.num_points)

        # plot
        ax.fill_between(x=time, y1=re, y2=np.zeros_like(time),
                        facecolor=self.style.wave_color[0], alpha=0.3,
                        edgecolor=self.style.wave_color[0], linewidth=1.5,
                        label='real part')
        ax.fill_between(x=time, y1=im, y2=np.zeros_like(time),
                        facecolor=self.style.wave_color[1], alpha=0.3,
                        edgecolor=self.style.wave_color[1], linewidth=1.5,
                        label='imaginary part')

        ax.set_xlim(0, pulse.duration * dt)
        if scale:
            ax.set_ylim(-1/scale, 1/scale)
        else:
            v_max = max(max(np.abs(re)), max(np.abs(im)))
            ax.set_ylim(-1.2 * v_max, 1.2 * v_max)

        bbox = ax.get_position()

        # This check is here for backwards compatibility. Before, the check was around
        # the suptitle line, however since the font style can take on a type of None
        # we need to unfortunately check both the type and the value of the object.
        if isinstance(self.style.title_font_size, int) and self.style.title_font_size > 0:
            figure.suptitle(str(pulse.name),
                            fontsize=self.style.title_font_size,
                            y=bbox.y1 + 0.02,
                            va='bottom')

        return figure


class ScheduleDrawer:
    """A class to create figure for schedule and channel."""

    def __init__(self, style: SchedStyle):
        """Create new figure.

        Args:
            style: Style sheet for pulse schedule visualization.
        """
        self.style = style or SchedStyle()

    def _build_channels(self, schedule: ScheduleComponent,
                        channels: List[Channel],
                        t0: int, tf: int,
                        show_framechange_channels: bool = True
                        ) -> Tuple[Dict[Channel, EventsOutputChannels],
                                   Dict[Channel, EventsOutputChannels],
                                   Dict[Channel, EventsOutputChannels]]:
        """Create event table of each pulse channels in the given schedule.

        Args:
            schedule: Schedule object to plot.
            channels: Channels to plot.
            t0: Start time of plot.
            tf: End time of plot.
            show_framechange_channels: Plot channels only with FrameChanges (ShiftPhase).

        Returns:
            channels: All channels.
            output_channels: All (D, M, U, A) channels.
            snapshot_channels: Snapshots.
        """
        # prepare waveform channels
        drive_channels = collections.OrderedDict()
        measure_channels = collections.OrderedDict()
        control_channels = collections.OrderedDict()
        acquire_channels = collections.OrderedDict()
        snapshot_channels = collections.OrderedDict()
        _channels = set()
        if show_framechange_channels:
            _channels.update(schedule.channels)
        # take channels that do not only contain framechanges
        else:
            for start_time, instruction in schedule.instructions:
                if not isinstance(instruction, (ShiftPhase, SetPhase)):
                    _channels.update(instruction.channels)

        _channels.update(channels)
        for chan in _channels:
            if isinstance(chan, DriveChannel):
                try:
                    drive_channels[chan] = EventsOutputChannels(t0, tf)
                except PulseError:
                    pass
            elif isinstance(chan, MeasureChannel):
                try:
                    measure_channels[chan] = EventsOutputChannels(t0, tf)
                except PulseError:
                    pass
            elif isinstance(chan, ControlChannel):
                try:
                    control_channels[chan] = EventsOutputChannels(t0, tf)
                except PulseError:
                    pass
            elif isinstance(chan, AcquireChannel):
                try:
                    acquire_channels[chan] = EventsOutputChannels(t0, tf)
                except PulseError:
                    pass
            elif isinstance(chan, SnapshotChannel):
                try:
                    snapshot_channels[chan] = EventsOutputChannels(t0, tf)
                except PulseError:
                    pass

        output_channels = {**drive_channels, **measure_channels,
                           **control_channels, **acquire_channels}
        channels = {**output_channels, **snapshot_channels}
        # sort by index then name to group qubits together.
        output_channels = collections.OrderedDict(sorted(output_channels.items(),
                                                         key=lambda x: (x[0].index, x[0].name)))
        channels = collections.OrderedDict(sorted(channels.items(),
                                                  key=lambda x: (x[0].index, x[0].name)))

        for start_time, instruction in schedule.instructions:
            for channel in instruction.channels:
                if channel in output_channels:
                    output_channels[channel].add_instruction(start_time, instruction)
                elif channel in snapshot_channels:
                    snapshot_channels[channel].add_instruction(start_time, instruction)
        return channels, output_channels, snapshot_channels

    @staticmethod
    def _scale_channels(output_channels: Dict[Channel, EventsOutputChannels],
                        scale: float,
                        channel_scales: Dict[Channel, float] = None,
                        channels: List[Channel] = None,
                        plot_all: bool = False) -> Dict[Channel, float]:
        """Count number of channels that contains any instruction to show
        and find scale factor of that channel.

        Args:
            output_channels: Event table of channels to show.
            scale: Global scale factor.
            channel_scales: Channel specific scale factors.
            channels: Specified channels to plot.
            plot_all: Plot empty channel.

        Returns:
            scale_dict: Scale factor of each channel.
        """
        # count numbers of valid waveform
        scale_dict = {chan: 0 for chan in output_channels.keys()}
        for channel, events in output_channels.items():
            v_max = 0
            if channels:
                if channel in channels:
                    waveform = events.waveform
                    v_max = max(v_max,
                                max(np.abs(np.real(waveform))),
                                max(np.abs(np.imag(waveform))))
                    events.enable = True
            else:
                if not events.is_empty() or plot_all:
                    waveform = events.waveform
                    v_max = max(v_max,
                                max(np.abs(np.real(waveform))),
                                max(np.abs(np.imag(waveform))))
                    events.enable = True

            scale_val = channel_scales.get(channel, scale)
            if not scale_val:
                # when input schedule is empty or comprises only frame changes,
                # we need to overwrite maximum amplitude by a value greater than zero,
                # otherwise auto axis scaling will fail with zero division.
                v_max = v_max or 1
                scale_dict[channel] = 1 / v_max
            else:
                scale_dict[channel] = scale_val

        return scale_dict

    def _draw_table(self, figure,
                    channels: Dict[Channel, EventsOutputChannels],
                    dt: float):
        """Draw event table if events exist.

        Args:
            figure (matpotlib.figure.Figure): Figure object
            channels: Dictionary of channel and event table
            dt: Time interval

        Returns:
            Tuple[matplotlib.axes.Axes]: Axis objects for table and canvas of pulses.
        """
        # create table
        table_data = []
        if self.style.use_table:
            for channel, events in channels.items():
                if events.enable:
                    table_data.extend(events.to_table(channel.name))
            table_data = sorted(table_data, key=lambda x: x[0])

        # plot table
        if table_data:
            # table area size
            ncols = self.style.table_columns
            nrows = int(np.ceil(len(table_data)/ncols))
            max_size = self.style.max_table_ratio * figure.get_size_inches()[1]
            max_rows = np.floor(max_size/self.style.fig_unit_h_table/ncols)
            nrows = int(min(nrows, max_rows))
            # don't overflow plot with table data
            table_data = table_data[:int(nrows*ncols)]
            # fig size
            h_table = nrows * self.style.fig_unit_h_table
            h_waves = (figure.get_size_inches()[1] - h_table)

            # create subplots
            gs = gridspec.GridSpec(2, 1, height_ratios=[h_table, h_waves], hspace=0)
            tb = plt.subplot(gs[0])
            ax = plt.subplot(gs[1])

            # configure each cell
            tb.axis('off')
            cell_value = [['' for _kk in range(ncols * 3)] for _jj in range(nrows)]
            cell_color = [self.style.table_color * ncols for _jj in range(nrows)]
            cell_width = [*([0.2, 0.2, 0.5] * ncols)]
            for ii, data in enumerate(table_data):
                # pylint: disable=unbalanced-tuple-unpacking
                r, c = np.unravel_index(ii, (nrows, ncols), order='f')
                # pylint: enable=unbalanced-tuple-unpacking
                time, ch_name, data_str = data
                # item
                cell_value[r][3 * c + 0] = 't = %s' % time * dt
                cell_value[r][3 * c + 1] = 'ch %s' % ch_name
                cell_value[r][3 * c + 2] = data_str
            table = tb.table(cellText=cell_value,
                             cellLoc='left',
                             rowLoc='center',
                             colWidths=cell_width,
                             bbox=[0, 0, 1, 1],
                             cellColours=cell_color)
            table.auto_set_font_size(False)
            table.set_fontsize = self.style.table_font_size
        else:
            tb = None
            ax = figure.add_subplot(111)

        return tb, ax

    @staticmethod
    def _draw_snapshots(ax,
                        snapshot_channels: Dict[Channel, EventsOutputChannels],
                        y0: float) -> None:
        """Draw snapshots to given mpl axis.

        Args:
            ax (matplotlib.axes.Axes): axis object to draw snapshots.
            snapshot_channels: Event table of snapshots.
            y0: vertical position to draw the snapshots.
        """
        for events in snapshot_channels.values():
            snapshots = events.snapshots
            if snapshots:
                for time in snapshots:
                    ax.annotate(s=u"\u25D8", xy=(time, y0), xytext=(time, y0+0.08),
                                arrowprops={'arrowstyle': 'wedge'}, ha='center')

    def _draw_framechanges(self, ax,
                           fcs: Dict[int, ShiftPhase],
                           y0: float) -> bool:
        """Draw frame change of given channel to given mpl axis.

        Args:
            ax (matplotlib.axes.Axes): axis object to draw frame changes.
            fcs: Event table of frame changes.
            y0: vertical position to draw the frame changes.
        """
        for time in fcs.keys():
            ax.text(x=time, y=y0, s=r'$\circlearrowleft$',
                    fontsize=self.style.icon_font_size,
                    ha='center', va='center')

    def _draw_frequency_changes(self, ax,
                                sf: Dict[int, SetFrequency],
                                y0: float) -> bool:
        """Draw set frequency of given channel to given mpl axis.

        Args:
            ax (matplotlib.axes.Axes): axis object to draw frame changes.
            sf: Event table of set frequency.
            y0: vertical position to draw the frame changes.
        """
        for time in sf.keys():
            ax.text(x=time, y=y0, s=r'$\leftrightsquigarrow$',
                    fontsize=self.style.icon_font_size,
                    ha='center', va='center', rotation=90)

    def _get_channel_color(self, channel: Channel) -> str:
        """Lookup table for waveform color.

        Args:
            channel: Type of channel.

        Return:
            Color code or name of color.
        """
        # choose color
        if isinstance(channel, DriveChannel):
            color = self.style.d_ch_color
        elif isinstance(channel, ControlChannel):
            color = self.style.u_ch_color
        elif isinstance(channel, MeasureChannel):
            color = self.style.m_ch_color
        elif isinstance(channel, AcquireChannel):
            color = self.style.a_ch_color
        else:
            color = 'black'
        return color

    @staticmethod
    def _prev_label_at_time(prev_labels: List[Dict[int, Union[Waveform, Acquire]]],
                            time: int) -> bool:
        """Check overlap of pulses with pervious channels.

        Args:
            prev_labels: List of labels in previous channels.
            time: Start time of current pulse instruction.

        Returns:
            `True` if current instruction overlaps with others.
        """
        for labels in prev_labels:
            for t0, (tf, _) in labels.items():
                if time in (t0, tf):
                    return True
        return False

    def _draw_labels(self, ax,
                     labels: Dict[int, Union[Waveform, Acquire]],
                     prev_labels: List[Dict[int, Union[Waveform, Acquire]]],
                     y0: float) -> None:
        """Draw label of pulse instructions on given mpl axis.

        Args:
            ax (matplotlib.axes.Axes): axis object to draw labels.
            labels: Pulse labels of channel.
            prev_labels: Pulse labels of previous channels.
            y0: vertical position to draw the labels.
        """
        for t0, (tf, cmd) in labels.items():
            if isinstance(cmd, Acquire):
                name = cmd.name if cmd.name else 'acquire'
            else:
                name = cmd.name

            ax.annotate(r'%s' % name,
                        xy=((t0+tf)//2, y0),
                        xytext=((t0+tf)//2, y0-0.07),
                        fontsize=self.style.label_font_size,
                        ha='center', va='center')

            linestyle = self.style.label_ch_linestyle
            alpha = self.style.label_ch_alpha
            color = self.style.label_ch_color

            if not self._prev_label_at_time(prev_labels, t0):
                ax.axvline(t0, -1, 1, color=color,
                           linestyle=linestyle, alpha=alpha)
            if not (self._prev_label_at_time(prev_labels, tf) or tf in labels):
                ax.axvline(tf, -1, 1, color=color,
                           linestyle=linestyle, alpha=alpha)

    def _draw_channels(self, ax,
                       output_channels: Dict[Channel, EventsOutputChannels],
                       interp_method: Callable,
                       t0: int, tf: int,
                       scale_dict: Dict[Channel, float],
                       label: bool = False,
                       framechange: bool = True,
                       frequencychange: bool = True) -> float:
        """Draw pulse instructions on given mpl axis.

        Args:
            ax (matplotlib.axes.Axes): axis object to draw pulses.
            output_channels: Event table of channels.
            interp_method: Callback function for waveform interpolation.
            t0: Start time of schedule.
            tf: End time of schedule.
            scale_dict: Scale factor for each channel.
            label: When set `True` draw labels.
            framechange: When set `True` draw frame change symbols.
            frequencychange: When set `True` draw frequency change symbols.

        Return:
            Value of final vertical axis of canvas.
        """
        y0 = 0
        prev_labels = []
        for channel, events in output_channels.items():
            if events.enable:
                # scaling value of this channel
                scale = 0.5 * scale_dict.get(channel, 0.5)
                # plot waveform
                waveform = events.waveform
                time = np.arange(t0, tf + 1, dtype=float)
                if waveform.any():
                    time, re, im = interp_method(time, waveform, self.style.num_points)
                else:
                    # when input schedule is empty or comprises only frame changes,
                    # we should avoid interpolation due to lack of data points.
                    # instead, it just returns vector of zero.
                    re, im = np.zeros_like(time), np.zeros_like(time)
                color = self._get_channel_color(channel)
                # Minimum amplitude scaled
                amp_min = scale * abs(min(0, np.nanmin(re), np.nanmin(im)))
                # scaling and offset
                re = scale * re + y0
                im = scale * im + y0
                offset = np.zeros_like(time) + y0
                # plot
                ax.fill_between(x=time, y1=re, y2=offset,
                                facecolor=color[0], alpha=0.3,
                                edgecolor=color[0], linewidth=1.5,
                                label='real part')
                ax.fill_between(x=time, y1=im, y2=offset,
                                facecolor=color[1], alpha=0.3,
                                edgecolor=color[1], linewidth=1.5,
                                label='imaginary part')
                ax.plot((t0, tf), (y0, y0), color='#000000', linewidth=1.0)

                # plot frame changes
                fcs = events.framechanges
                if fcs and framechange:
                    self._draw_framechanges(ax, fcs, y0)
                # plot frequency changes
                sf = events.frequencychanges
                if sf and frequencychange:
                    self._draw_frequency_changes(ax, sf, y0 + scale)
                # plot labels
                labels = events.labels
                if labels and label:
                    self._draw_labels(ax, labels, prev_labels, y0)
                prev_labels.append(labels)

            else:
                continue

            # plot label
            ax.text(x=t0, y=y0, s=channel.name,
                    fontsize=self.style.axis_font_size,
                    ha='right', va='center')
            # show scaling factor
            ax.text(x=t0, y=y0 - 0.1, s='x%.1f' % (2 * scale),
                    fontsize=0.7*self.style.axis_font_size,
                    ha='right', va='top')

            # change the y0 offset for removing spacing when a channel has negative values
            if self.style.remove_spacing:
                y0 -= 0.5 + amp_min
            else:
                y0 -= 1
        return y0

    def draw(self, schedule: ScheduleComponent,
             dt: float, interp_method: Callable,
             plot_range: Tuple[Union[int, float], Union[int, float]],
             scale: float = None,
             channel_scales: Dict[Channel, float] = None,
             plot_all: bool = True, table: bool = False,
             label: bool = False, framechange: bool = True,
             channels: List[Channel] = None,
             show_framechange_channels: bool = True):
        """Draw figure.

        Args:
            schedule: schedule object to plot.
            dt: Time interval of samples. Pulses are visualized in the unit of
                cycle time if not provided.
            interp_method: Interpolation function. See example.
                Interpolation is disabled in default.
                See `qiskit.visualization.pulse.interpolation` for more information.
            plot_range: A tuple of time range to plot.
            scale: Scaling of waveform amplitude. Pulses are automatically
                scaled channel by channel if not provided.
            channel_scales: Dictionary of scale factor for specific channels.
                Scale of channels not specified here is overwritten by `scale`.
            plot_all: When set `True` plot empty channels.
            table: When set `True` draw event table for supported commands.
            label: When set `True` draw label for individual instructions.
            framechange: When set `True` draw framechange indicators.
            channels: A list of channel names to plot.
                All non-empty channels are shown if not provided.
            show_framechange_channels: When set `True` plot channels
                with only framechange instructions.

        Returns:
            matplotlib.figure.Figure: A matplotlib figure object for the pulse envelope.

        Raises:
            VisualizationError: When schedule cannot be drawn
        """
        figure = plt.figure(dpi=self.style.dpi, figsize=self.style.figsize)

        if channels is None:
            channels = []
        interp_method = interp_method or step_wise

        if channel_scales is None:
            channel_scales = {}

        # setup plot range
        if plot_range:
            t0 = int(np.floor(plot_range[0]))
            tf = int(np.floor(plot_range[1]))
        else:
            t0 = 0
            # when input schedule is empty or comprises only frame changes,
            # we need to overwrite pulse duration by an integer greater than zero,
            # otherwise waveform returns empty array and matplotlib will be crashed.
            if channels:
                tf = schedule.ch_duration(*channels)
            else:
                tf = schedule.stop_time
            tf = tf or 1

        # prepare waveform channels
        (schedule_channels, output_channels,
         snapshot_channels) = self._build_channels(schedule, channels, t0, tf,
                                                   show_framechange_channels)

        # count numbers of valid waveform
        scale_dict = self._scale_channels(output_channels,
                                          scale=scale,
                                          channel_scales=channel_scales,
                                          channels=channels,
                                          plot_all=plot_all)

        if table:
            tb, ax = self._draw_table(figure, schedule_channels, dt)
        else:
            tb = None
            ax = figure.add_subplot(111)

        ax.set_facecolor(self.style.bg_color)

        y0 = self._draw_channels(ax, output_channels, interp_method,
                                 t0, tf, scale_dict, label=label,
                                 framechange=framechange)

        y_ub = 0.5 + self.style.vertical_span
        y_lb = y0 + 0.5 - self.style.vertical_span

        self._draw_snapshots(ax, snapshot_channels, y_lb)

        ax.set_xlim(t0, tf)
        tick_labels = np.linspace(t0, tf, 5)
        ax.set_xticks(tick_labels)
        ax.set_xticklabels([self.style.axis_formatter % label for label in tick_labels * dt],
                           fontsize=self.style.axis_font_size)
        ax.set_ylim(y_lb, y_ub)
        ax.set_yticklabels([])

        if tb is not None:
            bbox = tb.get_position()
        else:
            bbox = ax.get_position()

        # This check is here for backwards compatibility. Before, the check was around
        # the suptitle line, however since the font style can take on a type of None
        # we need to unfortunately check both the type and the value of the object.
        if isinstance(self.style.title_font_size, int) and self.style.title_font_size > 0:
            figure.suptitle(str(schedule.name),
                            fontsize=self.style.title_font_size,
                            y=bbox.y1 + 0.02,
                            va='bottom')

        return figure
