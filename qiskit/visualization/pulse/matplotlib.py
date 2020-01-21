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
import warnings

import numpy as np

try:
    from matplotlib import pyplot as plt, gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from qiskit.visualization.pulse.qcstyle import PulseStyle, SchedStyle
from qiskit.visualization.pulse import interpolation
from qiskit.pulse.channels import (DriveChannel, ControlChannel,
                                   MeasureChannel, AcquireChannel,
                                   SnapshotChannel)
from qiskit.pulse import (SamplePulse, FrameChange, PersistentValue, Snapshot,
                          Acquire, PulseError, ParametricPulse)
from qiskit.pulse.commands.frame_change import FrameChangeInstruction


class EventsOutputChannels:
    """Pulse dataset for channel."""

    def __init__(self, t0, tf):
        """Create new channel dataset.

        Args:
            t0 (int): starting time of plot
            tf (int): ending time of plot
        """
        self.pulses = {}
        self.t0 = t0
        self.tf = tf

        self._waveform = None
        self._framechanges = None
        self._conditionals = None
        self._snapshots = None
        self._labels = None
        self.enable = False

    def add_instruction(self, start_time, pulse):
        """Add new pulse instruction to channel.

        Args:
            start_time (int): Starting time of instruction
            pulse (Instruction): Instruction object to be added
        """

        if start_time in self.pulses.keys():
            self.pulses[start_time].append(pulse.command)
        else:
            self.pulses[start_time] = [pulse.command]

    @property
    def waveform(self):
        """Get waveform."""
        if self._waveform is None:
            self._build_waveform()

        return self._waveform[self.t0:self.tf]

    @property
    def framechanges(self):
        """Get frame changes."""
        if self._framechanges is None:
            self._build_waveform()

        return self._trim(self._framechanges)

    @property
    def conditionals(self):
        """Get conditionals."""
        if self._conditionals is None:
            self._build_waveform()

        return self._trim(self._conditionals)

    @property
    def snapshots(self):
        """Get snapshots."""
        if self._snapshots is None:
            self._build_waveform()

        return self._trim(self._snapshots)

    @property
    def labels(self):
        """Get labels."""
        if self._labels is None:
            self._build_waveform()

        return self._trim(self._labels)

    def is_empty(self):
        """Return if pulse is empty.

        Returns:
            bool: if the channel has nothing to plot
        """
        if any(self.waveform) or self.framechanges or self.conditionals or self.snapshots:
            return False

        return True

    def to_table(self, name):
        """Get table contains.

        Args:
            name (str): name of channel

        Returns:
            dict: dictionary of events in the channel
        """
        time_event = []

        framechanges = self.framechanges
        conditionals = self.conditionals
        snapshots = self.snapshots

        for key, val in framechanges.items():
            data_str = 'framechange: %.2f' % val
            time_event.append((key, name, data_str))
        for key, val in conditionals.items():
            data_str = 'conditional, %s' % val
            time_event.append((key, name, data_str))
        for key, val in snapshots.items():
            data_str = 'snapshot: %s' % val
            time_event.append((key, name, data_str))

        return time_event

    def _build_waveform(self):
        """Create waveform from stored pulses.
        """
        self._framechanges = {}
        self._conditionals = {}
        self._snapshots = {}
        self._labels = {}
        fc = 0
        pv = np.zeros(self.tf + 1, dtype=np.complex128)
        wf = np.zeros(self.tf + 1, dtype=np.complex128)
        last_pv = None
        for time, commands in sorted(self.pulses.items()):
            if time > self.tf:
                break
            tmp_fc = 0
            for command in commands:
                if isinstance(command, FrameChange):
                    tmp_fc += command.phase
                    pv[time:] = 0
                elif isinstance(command, Snapshot):
                    self._snapshots[time] = command.name
            if tmp_fc != 0:
                self._framechanges[time] = tmp_fc
                fc += tmp_fc
            for command in commands:
                if isinstance(command, PersistentValue):
                    pv[time:] = np.exp(1j*fc) * command.value
                    last_pv = (time, command)
                    break

            for command in commands:
                duration = command.duration
                tf = min(time + duration, self.tf)
                if isinstance(command, ParametricPulse):
                    command = command.get_sample_pulse()
                if isinstance(command, SamplePulse):
                    wf[time:tf] = np.exp(1j*fc) * command.samples[:tf-time]
                    pv[time:] = 0
                    self._labels[time] = (tf, command)
                    if last_pv is not None:
                        pv_cmd = last_pv[1]
                        self._labels[last_pv[0]] = (time, pv_cmd)
                        last_pv = None

                elif isinstance(command, Acquire):
                    wf[time:tf] = np.ones(tf - time)
                    self._labels[time] = (tf, command)
        self._waveform = wf + pv

    def _trim(self, events):
        """Return events during given `time_range`.

        Args:
            events (dict): time and operation of events

        Returns:
            dict: dictionary of events within the time
        """
        events_in_time_range = {}

        for k, v in events.items():
            if self.t0 <= k <= self.tf:
                events_in_time_range[k] = v

        return events_in_time_range


class SamplePulseDrawer:
    """A class to create figure for sample pulse."""

    def __init__(self, style):
        """Create new figure.

        Args:
            style (PulseStyle): style sheet
        """
        self.style = style or PulseStyle()

    def draw(self, pulse, dt, interp_method, scale=1, scaling=None):
        """Draw figure.

        Args:
            pulse (SamplePulse): SamplePulse to draw
            dt (float): time interval
            interp_method (Callable): interpolation function
                See `qiskit.visualization.interpolation` for more information
            scale (float): Relative visual scaling of waveform amplitudes
            scaling (float): Deprecated, see `scale`

        Returns:
            matplotlib.figure: A matplotlib figure object of the pulse envelope
        """
        if scaling is not None:
            warnings.warn('The parameter "scaling" is being replaced by "scale"',
                          DeprecationWarning, 3)
            scale = scaling
        figure = plt.figure()

        interp_method = interp_method or interpolation.step_wise

        figure.set_size_inches(self.style.figsize[0], self.style.figsize[1])
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

        return figure


class ScheduleDrawer:
    """A class to create figure for schedule and channel."""

    def __init__(self, style):
        """Create new figure.

        Args:
            style (SchedStyle): style sheet
        """
        self.style = style or SchedStyle()

    def _build_channels(self, schedule, channels, t0, tf, show_framechange_channels=True):
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
                if not isinstance(instruction, FrameChangeInstruction):
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
        channels = {**output_channels, **acquire_channels, **snapshot_channels}
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

    def _count_valid_waveforms(self, output_channels, scale=1, channels=None,
                               plot_all=False, scaling=None):
        if scaling is not None:
            warnings.warn('The parameter "scaling" is being replaced by "scale"',
                          DeprecationWarning, 3)
            scale = scaling
        # count numbers of valid waveform
        n_valid_waveform = 0
        v_max = 0
        for channel, events in output_channels.items():
            if channels:
                if channel in channels:
                    waveform = events.waveform
                    v_max = max(v_max,
                                max(np.abs(np.real(waveform))),
                                max(np.abs(np.imag(waveform))))
                    n_valid_waveform += 1
                    events.enable = True
            else:
                if not events.is_empty() or plot_all:
                    waveform = events.waveform
                    v_max = max(v_max,
                                max(np.abs(np.real(waveform))),
                                max(np.abs(np.imag(waveform))))
                    n_valid_waveform += 1
                    events.enable = True

        # when input schedule is empty or comprises only frame changes,
        # we need to overwrite maximum amplitude by a value greater than zero,
        # otherwise auto axis scaling will fail with zero division.
        v_max = v_max or 1

        if scale:
            v_max = 0.5 * scale
        else:
            v_max = 0.5 / (v_max)

        return n_valid_waveform, v_max

    # pylint: disable=unused-argument
    def _draw_table(self, figure, channels, dt, n_valid_waveform):
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
            max_size = self.style.max_table_ratio * self.style.figsize[1]
            max_rows = np.floor(max_size/self.style.fig_unit_h_table/ncols)
            nrows = int(min(nrows, max_rows))
            # don't overflow plot with table data
            table_data = table_data[:int(nrows*ncols)]
            # fig size
            h_table = nrows * self.style.fig_unit_h_table
            h_waves = (self.style.figsize[1] - h_table)

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
            ax = figure.add_subplot(111)

        figure.set_size_inches(self.style.figsize[0], self.style.figsize[1])

        return ax

    def _draw_snapshots(self, ax, snapshot_channels, dt, y0):
        for events in snapshot_channels.values():
            snapshots = events.snapshots
            if snapshots:
                for time in snapshots:
                    ax.annotate(s=u"\u25D8", xy=(time*dt, y0), xytext=(time*dt, y0+0.08),
                                arrowprops={'arrowstyle': 'wedge'}, ha='center')

    def _draw_framechanges(self, ax, fcs, dt, y0):
        framechanges_present = True
        for time in fcs.keys():
            ax.text(x=time*dt, y=y0, s=r'$\circlearrowleft$',
                    fontsize=self.style.icon_font_size,
                    ha='center', va='center')
        return framechanges_present

    def _get_channel_color(self, channel):
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

    def _prev_label_at_time(self, prev_labels, time):
        for _, labels in enumerate(prev_labels):
            for t0, (tf, _) in labels.items():
                if time in (t0, tf):
                    return True
        return False

    def _draw_labels(self, ax, labels, prev_labels, dt, y0):
        for t0, (tf, cmd) in labels.items():
            if isinstance(cmd, PersistentValue):
                name = cmd.name if cmd.name else 'pv'
            elif isinstance(cmd, Acquire):
                name = cmd.name if cmd.name else 'acquire'
            else:
                name = cmd.name

            ax.annotate(r'%s' % name,
                        xy=((t0+tf)//2*dt, y0),
                        xytext=((t0+tf)//2*dt, y0-0.07),
                        fontsize=self.style.label_font_size,
                        ha='center', va='center')

            linestyle = self.style.label_ch_linestyle
            alpha = self.style.label_ch_alpha
            color = self.style.label_ch_color

            if not self._prev_label_at_time(prev_labels, t0):
                ax.axvline(t0*dt, -1, 1, color=color,
                           linestyle=linestyle, alpha=alpha)
            if not (self._prev_label_at_time(prev_labels, tf) or tf in labels):
                ax.axvline(tf*dt, -1, 1, color=color,
                           linestyle=linestyle, alpha=alpha)

    def _draw_channels(self, ax, output_channels, interp_method, t0, tf, dt, v_max,
                       label=False, framechange=True):
        y0 = 0
        prev_labels = []
        for channel, events in output_channels.items():
            if events.enable:
                # plot waveform
                waveform = events.waveform
                time = np.arange(t0, tf + 1, dtype=float) * dt
                if waveform.any():
                    time, re, im = interp_method(time, waveform, self.style.num_points)
                else:
                    # when input schedule is empty or comprises only frame changes,
                    # we should avoid interpolation due to lack of data points.
                    # instead, it just returns vector of zero.
                    re, im = np.zeros_like(time), np.zeros_like(time)
                color = self._get_channel_color(channel)
                # Minimum amplitude scaled
                amp_min = v_max * abs(min(0, np.nanmin(re), np.nanmin(im)))
                # scaling and offset
                re = v_max * re + y0
                im = v_max * im + y0
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
                    self._draw_framechanges(ax, fcs, dt, y0)
                # plot labels
                labels = events.labels
                if labels and label:
                    self._draw_labels(ax, labels, prev_labels, dt, y0)
                prev_labels.append(labels)

            else:
                continue

            # plot label
            ax.text(x=0, y=y0, s=channel.name,
                    fontsize=self.style.axis_font_size,
                    ha='right', va='center')

            # change the y0 offset for removing spacing when a channel has negative values
            if self.style.remove_spacing:
                y0 -= 0.5 + amp_min
            else:
                y0 -= 1
        return y0

    def draw(self, schedule, dt, interp_method, plot_range,
             scale=None, channels_to_plot=None, plot_all=True,
             table=True, label=False, framechange=True,
             scaling=None, channels=None,
             show_framechange_channels=True):
        """Draw figure.

        Args:
            schedule (ScheduleComponent): Schedule to draw
            dt (float): time interval
            interp_method (Callable): interpolation function
                See `qiskit.visualization.interpolation` for more information
            plot_range (tuple[float]): plot range
            scale (float): Relative visual scaling of waveform amplitudes
            channels_to_plot (list[OutputChannel]): deprecated, see `channels`
            plot_all (bool): if plot all channels even it is empty
            table (bool): Draw event table
            label (bool): Label individual instructions
            framechange (bool): Add framechange indicators
            scaling (float): Deprecated, see `scale`
            channels (list[OutputChannel]): channels to draw
            show_framechange_channels (bool): Plot channels with only framechanges

        Returns:
            matplotlib.figure: A matplotlib figure object for the pulse schedule
        Raises:
            VisualizationError: when schedule cannot be drawn
        """
        if scaling is not None:
            warnings.warn('The parameter "scaling" is being replaced by "scale"',
                          DeprecationWarning, 3)
            scale = scaling
        figure = plt.figure()

        if channels_to_plot is not None:
            warnings.warn('The parameter "channels_to_plot" is being replaced by "channels"',
                          DeprecationWarning, 3)
            channels = channels_to_plot

        if channels is None:
            channels = []
        interp_method = interp_method or interpolation.step_wise

        # setup plot range
        if plot_range:
            t0 = int(np.floor(plot_range[0]/dt))
            tf = int(np.floor(plot_range[1]/dt))
        else:
            t0 = 0
            # when input schedule is empty or comprises only frame changes,
            # we need to overwrite pulse duration by an integer greater than zero,
            # otherwise waveform returns empty array and matplotlib will be crashed.
            if channels:
                tf = schedule.timeslots.ch_duration(*channels)
            else:
                tf = schedule.stop_time
            tf = tf or 1

        # prepare waveform channels
        (schedule_channels, output_channels,
         snapshot_channels) = self._build_channels(schedule, channels, t0, tf,
                                                   show_framechange_channels)

        # count numbers of valid waveform

        n_valid_waveform, v_max = self._count_valid_waveforms(output_channels,
                                                              scale=scale,
                                                              channels=channels,
                                                              plot_all=plot_all)

        if table:
            ax = self._draw_table(figure, schedule_channels, dt, n_valid_waveform)

        else:
            ax = figure.add_subplot(111)
            figure.set_size_inches(self.style.figsize[0], self.style.figsize[1])

        ax.set_facecolor(self.style.bg_color)

        y0 = self._draw_channels(ax, output_channels, interp_method,
                                 t0, tf, dt, v_max, label=label,
                                 framechange=framechange)

        self._draw_snapshots(ax, snapshot_channels, dt, y0)

        ax.set_xlim(t0 * dt, tf * dt)
        ax.set_ylim(y0, 1)
        ax.set_yticklabels([])

        return figure
