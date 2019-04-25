# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
mpl pulse visualization.
"""

import logging
from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt, gridspec

from qiskit.pulse import SamplePulse, FrameChange, PersistentValue, Schedule
from qiskit.pulse.channels import DriveChannel, ControlChannel, MeasureChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.interpolation import cubic_spline
from qiskit.visualization.qcstyle import OPStylePulse, OPStyleSched

logger = logging.getLogger(__name__)


def pulse_drawer(data, device=None, dt=1, style=None, filename=None,
                 interp_method=None, scaling=None, channels_to_plot=None,
                 plot_all=False, plot_range=None, interactive=False):
    """Plot the interpolated envelope of pulse

    Args:
        data (Schedule or SamplePulse): Data to plot.
        device (DeviceSpecification): Device information to organize channels.
        dt (float): Time interval of samples.
        style (OPStylePulse or OPStyleSched): A style sheet to configure plot appearance.
        filename (str): Name required to save pulse image.
        interp_method (Callable): A function for interpolation.
        scaling (float): scaling of waveform amplitude.
        channels_to_plot (list): A list of channel names to plot.
        plot_all (bool): Plot empty channels.
        plot_range (tuple): A tuple of time range to plot.
        interactive (bool): When set true show the circuit in a new window
            (this depends on the matplotlib backend being used supporting this).
    Returns:
        matplotlib.figure: A matplotlib figure object for the pulse envelope.
    Raises:
        VisualizationError: when invalid data is given or lack of information.
    """
    if isinstance(data, SamplePulse):
        drawer = SamplePulseDrawer(style=style)
        image = drawer.draw(pulse_obj=data, dt=dt,
                            interp_method=interp_method, scaling=scaling)
    elif isinstance(data, Schedule):
        if not device:
            raise VisualizationError('Schedule visualizer needs device information.')
        drawer = ScheduleDrawer(device=device, style=style)
        image = drawer.draw(pulse_obj=data, dt=dt,
                            interp_method=interp_method, scaling=scaling,
                            plot_range=plot_range, channels_to_plot=channels_to_plot,
                            plot_all=plot_all)
    else:
        raise VisualizationError('This data cannot be visualized.')

    if filename:
        image.savefig(filename, dpi=drawer.style.dpi, bbox_inches='tight')

    plt.close(image)

    if image and interactive:
        image.show()
    return image


class EventsOutputChannels:
    """Pulse dataset for channel."""

    def __init__(self, t0, tf):
        """Create new channel dataset.

        Args:
            t0 (int): starting time of plot.
            tf (int): ending time of plot.
        """
        self.pulses = {}
        self.t0 = t0
        self.tf = tf

        self._waveform = None
        self._frame_change = None
        self._conditionals = None

        self.enable = False

    def add_instruction(self, pulse):
        """Add new pulse instruction to channel.

        Args:
            pulse (Instruction): Instruction object to be added.
        """

        if pulse.start_time in self.pulses.keys():
            self.pulses[pulse.start_time].append(pulse.command)
        else:
            self.pulses[pulse.start_time] = [pulse.command]

    @property
    def waveform(self):
        """Get waveform.
        """
        if self._waveform is None:
            self._build_waveform()

        return self._waveform[self.t0:self.tf]

    @property
    def frame_change(self):
        """Get frame changes.
        """
        if self._frame_change is None:
            self._build_waveform()

        return self._trim(self._frame_change)

    @property
    def conditionals(self):
        """Get conditionals.
        """
        if self._conditionals is None:
            self._build_waveform()

        return self._trim(self._conditionals)

    def is_empty(self):
        """Return if pulse is empty.

        Returns:
            bool: if the channel has nothing to plot.
        """
        if any(self.waveform) or self.frame_change:
            return False

        return True

    def to_table(self, name):
        """Get table contains.

        Args:
            name (str): name of channel.

        Returns:
            dict: dictionary of events in the channel.
        """
        time_event = []

        fc_pulses = self.frame_change
        conditionals = self.conditionals

        for key, val in fc_pulses.items():
            data_str = 'FrameChange, %.2f' % val
            time_event.append((key, name, data_str))
        for key, val in conditionals.items():
            data_str = 'Conditional, %s' % val
            time_event.append((key, name, data_str))

        return time_event

    def _build_waveform(self):
        """Create waveform from stored pulses.
        """
        self._frame_change = {}
        self._conditionals = {}

        fc = 0
        pv = np.zeros(self.tf + 1, dtype=np.complex128)
        wf = np.zeros(self.tf + 1, dtype=np.complex128)
        for time, commands in sorted(self.pulses.items()):
            if time > self.tf:
                break
            tmp_fc = 0
            for command in commands:
                if isinstance(command, FrameChange):
                    tmp_fc += command.phase
                    pv[time:] = 0
            if tmp_fc != 0:
                self._frame_change[time] = tmp_fc
                fc += tmp_fc
            for command in commands:
                if isinstance(command, PersistentValue):
                    pv[time:] = np.exp(1j*fc) * command.value
                    break
            for command in commands:
                if isinstance(command, SamplePulse):
                    tf = min(time + command.duration, self.tf)
                    wf[time:tf] = np.exp(1j*fc) * command.samples[:tf-time]
                    pv[time:] = 0

        self._waveform = wf + pv

    def _trim(self, events):
        """Return events during given `time_range`.

        Args:
            events (dict): time and operation of events.

        Returns:
            dict: dictionary of events within the time.
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
            style (OPStylePulse): style sheet.
        """
        self.style = style or OPStylePulse()

    def draw(self, pulse_obj, dt, interp_method, scaling):
        """Draw figure.
        Args:
            pulse_obj (SamplePulse): SamplePulse to draw.
            dt (float): time interval.
            interp_method (Callable): interpolation function.
            scaling (float): scaling of waveform amplitude.

        Returns:
            matplotlib.figure: A matplotlib figure object of the pulse envelope.
        """
        figure = plt.figure()

        interp_method = interp_method or cubic_spline

        figure.set_size_inches(self.style.fig_w, self.style.fig_h)
        ax = figure.add_subplot(111)
        ax.set_facecolor(self.style.bg_color)

        samples = pulse_obj.samples
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

        ax.set_xlim(0, pulse_obj.duration * dt)
        if scaling:
            ax.set_ylim(-scaling, scaling)
        else:
            v_max = max(max(np.abs(re)), max(np.abs(im)))
            ax.set_ylim(-1.2 * v_max, 1.2 * v_max)

        return figure


class ScheduleDrawer:
    """A class to create figure for schedule and channel."""

    def __init__(self, device, style):
        """Create new figure.

        Args:
            device (DeviceSpecification): configuration of device.
            style (OPStyleSched): style sheet.
        """
        self.device = device
        self.style = style or OPStyleSched()

    def draw(self, pulse_obj, dt, interp_method, scaling,
             plot_range, channels_to_plot, plot_all):
        """Draw figure.
        Args:
            pulse_obj (Schedule): Schedule to draw.
            dt (float): time interval.
            interp_method (Callable): interpolation function.
            scaling (float): scaling of waveform amplitude.
            plot_range (tuple[float]): plot range.
            channels_to_plot (list[OutputChannel]): channels to draw.
            plot_all (bool): if plot all channels even it is empty.

        Returns:
            matplotlib.figure: A matplotlib figure object for the pulse schedule.

        Raises:
            VisualizationError: when schedule cannot be drawn.
        """
        figure = plt.figure()

        if not channels_to_plot:
            channels_to_plot = []
        interp_method = interp_method or cubic_spline

        # setup plot range
        if plot_range:
            t0 = int(np.floor(plot_range[0]/dt))
            tf = int(np.floor(plot_range[1]/dt))
        else:
            t0 = 0
            tf = pulse_obj.stop_time

        # prepare waveform channels
        channels = OrderedDict()
        for q in self.device.q:
            try:
                channels[q.drive] = EventsOutputChannels(t0, tf)
            except PulseError:
                pass
            try:
                channels[q.control] = EventsOutputChannels(t0, tf)
            except PulseError:
                pass
            try:
                channels[q.measure] = EventsOutputChannels(t0, tf)
            except PulseError:
                pass

        for instruction in pulse_obj.flat_instruction_sequence():
            for channel in instruction.channels:
                if channel in channels:
                    channels[instruction.channel].add_instruction(instruction)

        # count numbers of valid waveform
        n_valid_waveform = 0
        v_max = 0
        for channel, events in channels.items():
            if channels_to_plot:
                if channel in channels_to_plot:
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
        if scaling:
            v_max = 0.5 / scaling
        else:
            v_max = 0.5 / (1.2 * v_max)

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

            # fig size
            h_table = nrows * self.style.fig_unit_h_table
            h_waves = n_valid_waveform * self.style.fig_unit_h_waveform
            fig_h = h_table + h_waves

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
                r, c = np.unravel_index(ii, (nrows, ncols), order='f')
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
            fig_h = n_valid_waveform * self.style.fig_unit_h_waveform
            ax = figure.add_subplot(111)

        figure.set_size_inches(self.style.fig_w, fig_h)
        ax.set_facecolor = self.style.bg_color

        y0 = 0
        for channel, events in channels.items():
            if events.enable:
                # plot waveform
                waveform = events.waveform
                time = np.arange(t0, tf + 1, dtype=float) * dt
                time, re, im = interp_method(time, waveform, self.style.num_points)
                # choose color
                if isinstance(channel, DriveChannel):
                    color = self.style.d_ch_color
                elif isinstance(channel, ControlChannel):
                    color = self.style.u_ch_color
                elif isinstance(channel, MeasureChannel):
                    color = self.style.m_ch_color
                else:
                    raise VisualizationError('Ch %s cannot be drawn.' % channel.name)
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
                fcs = events.frame_change
                if fcs:
                    for time in fcs.keys():
                        ax.text(x=time*dt, y=y0, s=r'$\circlearrowleft$',
                                fontsize=self.style.label_font_size,
                                ha='center', va='center')
                # plot label
                ax.text(x=0, y=y0, s=channel.name,
                        fontsize=self.style.label_font_size,
                        ha='right', va='center')
            else:
                continue
            y0 -= 1

        ax.set_xlim(t0 * dt, tf * dt)
        ax.set_ylim(y0, 1)
        ax.set_yticklabels([])

        return figure
