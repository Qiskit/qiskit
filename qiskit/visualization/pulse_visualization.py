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
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np

from qiskit.pulse import SamplePulse, Schedule, DeviceSpecification
from qiskit.pulse import commands
from qiskit.pulse.channels import DriveChannel, ControlChannel, MeasureChannel
from qiskit.tools.visualization import exceptions
from qiskit.visualization.interpolation import cubic_spline
from qiskit.visualization.qcstyle import OPStylePulse, OPStyleSched
from qiskit.pulse.exceptions import PulseError

try:
    from matplotlib import pyplot as plt, gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


def pulse_drawer(data, dt=1, device=None, style=None, filename=None,
                 interp_method=None, channels_to_plot=None,
                 plot_all=False, plot_range=None, scaling=None,
                 interactive=False):
    """Plot the interpolated envelope of pulse

    Args:
        data (Schedule or SamplePulse): Data to plot.
        dt (float): Time interval of samples.
        device (DeviceSpecification): Device information to organize channels.
        filename (str): Name required to save pulse image.
        interp_method (Callable): A function for interpolation.
        style (dict): A style sheet to configure plot appearance.
        channels_to_plot (list): A list of channel names to plot.
        plot_all (bool): Plot empty channels.
        plot_range (tuple): A tuple of time range to plot.
        scaling (float): scaling of waveform amplitude.
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
            raise exceptions.VisualizationError('Schedule visualizer needs device information.')
        drawer = ScheduleDrawer(device=device, style=style)
        image = drawer.draw(pulse_obj=data, dt=dt, plot_range=plot_range,
                            channels_to_plot=channels_to_plot,
                            plot_all=plot_all, interp_method=interp_method,
                            scaling=scaling)
    else:
        raise exceptions.VisualizationError('This data cannot be visualized.')

    if filename:
        image.savefig(filename, dpi=drawer.style.dpi, bbox_inches='tight')

    plt.close(image)

    if image and interactive:
        image.show()
    return image


class EventsOutputChannels:
    """Pulse dataset for channel."""

    def __init__(self, duration):
        """Create new channel dataset.

        Args:
            duration (int):
        """

        self.samples = np.zeros(duration + 1, dtype=np.complex128)
        self.fc_pulses = {}
        self.pv_pulses = {}
        self.conditionals = {}

        self.all_events = set()

        self.enable = False

    def add_instruction(self, pulse):
        """Add new pulse instruction to channel.

        Args:
            pulse (Instruction): Instruction object to be added.
        """
        if isinstance(pulse, commands.DriveInstruction):
            self.samples[pulse.start_time:pulse.stop_time] = pulse.command.samples
        elif isinstance(pulse, commands.FrameChangeInstruction):
            if pulse.start_time in self.fc_pulses.keys():
                self.fc_pulses[pulse.start_time] += pulse.command.phase
            else:
                self.fc_pulses[pulse.start_time] = pulse.command.phase
        elif isinstance(pulse, commands.PersistentValueInstruction):
            if pulse.start_time not in self.pv_pulses.keys():
                self.pv_pulses[pulse.start_time] = pulse.command.value
        else:
            return

        self.all_events.add(pulse.start_time)

    def get_waveform(self):
        """Get waveform.
        """
        fc_t = np.ones_like(self.samples)
        pv_t = np.zeros_like(self.samples)

        # sort fc by time index
        fcs = sorted(self.fc_pulses.items(), key=lambda x: x[0])
        for t0, val in fcs:
            fc_t[t0:] *= np.exp(1j * val)

        # sort pv by time index
        pvs = sorted(self.pv_pulses.items(), key=lambda x: x[0])
        _all_events = np.array(list(self.all_events))
        for t0, val in pvs:
            next_ts = _all_events[_all_events >= t0]
            if len(next_ts):
                pv_t[t0:min(next_ts)] = val
            else:
                pv_t[t0:] = val

        return fc_t * (self.samples + pv_t)

    def get_framechange(self, t0, tf):
        """Get frame changes to draw symbols.

        Args:
            name: name of channel.
            t0 (int): starting time of plot
            tf (int): ending time of plot

        Returns:
            dict: dictionary of events in the channel.
        """
        return self.trim(self.fc_pulses, t0, tf)

    def is_empty(self, t0, tf):
        """Return if pulse is empty.

        Args:
            t0 (int): starting time of plot
            tf (int): ending time of plot

        Returns:
            bool: if the channel has nothing to plot.
        """
        waveform = self.get_waveform()[t0:tf + 1]
        fc_pulses = self.trim(self.fc_pulses, t0, tf)

        if any(waveform) or len(fc_pulses):
            return False

        return True

    def to_table(self, name, t0, tf):
        """Get table contains.

        Args:
            name: name of channel.
            t0 (int): starting time of plot
            tf (int): ending time of plot

        Returns:
            dict: dictionary of events in the channel.
        """
        time_event = []

        fc_pulses = self.trim(self.fc_pulses, t0, tf)
        conditionals = self.trim(self.conditionals, t0, tf)

        for key, val in fc_pulses.items():
            data_str = 'FrameChange, %.2f' % val
            time_event.append((key, name, data_str))
        for key, val in conditionals.items():
            data_str = 'Conditional, %s' % val
            time_event.append((key, name, data_str))

        return time_event

    @staticmethod
    def trim(events, t0, tf):
        """Return events during given `time_range`.

        Args:
            events (dict): time and operation of events.
            t0 (int): starting time of plot
            tf (int): ending time of plot

        Returns:
            dict: dictionary of events within the time.
        """
        events_in_timerange = {}

        for k, v in events.items():
            if t0 <= k <= tf:
                events_in_timerange[k] = v

        return events_in_timerange


class OpenPulseDrawer(metaclass=ABCMeta):
    """Common interface for OpenPulse drawer."""

    @abstractmethod
    def draw(self, pulse_obj, dt, interp_method, scaling):
        """Draw OpenPulse waveform.

        Args:
            pulse_obj (ScheduleComponent): waveform data.
            dt (float): time interval.
            interp_method (Callable): interpolation function.
            scaling (float): scaling of waveform amplitude.
        """
        pass


class SamplePulseDrawer(OpenPulseDrawer):
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
        """
        figure = plt.figure()

        interp_method = interp_method or cubic_spline

        figure.set_size_inches(self.style.fig_w, self.style.fig_h)
        ax = figure.add_subplot(111)
        ax.set_facecolor(self.style.bg_color)

        samples = pulse_obj.samples
        time, re, im = interp_method(samples, dt, self.style.num_points)

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


class ScheduleDrawer(OpenPulseDrawer):
    """A class to create figure for schedule and channel."""

    def __init__(self, device, style):
        """Create new figure.

        Args:
            device (DeviceSpecification): configuration of device.
            style (OPStyleSched): style sheet.
        """
        self.device = device
        self.style = style or OPStyleSched()

    def draw(self, pulse_obj, dt, plot_range, channels_to_plot, plot_all,
             interp_method, scaling):
        """Draw figure.
        Args:
            pulse_obj (Schedule): Schedule to draw.
            dt (float): time interval.
            plot_range (tuple[float]): plot range.
            channels_to_plot (list[OutputChannel]): channels to draw.
            plot_all (bool): if plot all channels even it is empty.
            interp_method (Callable): interpolation function.
            scaling (float): scaling of waveform amplitude.
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
                channels[q.drive] = EventsOutputChannels(duration=tf)
            except PulseError:
                pass
            try:
                channels[q.control] = EventsOutputChannels(duration=tf)
            except PulseError:
                pass
            try:
                channels[q.measure] = EventsOutputChannels(duration=tf)
            except PulseError:
                pass

        for instruction in pulse_obj.flat_instruction_sequence():
            channels[instruction.channel].add_instruction(instruction)

        # count numbers of valid waveform
        n_valid_waveform = 0
        v_max = 0
        for channel, events in channels.items():
            if not events.is_empty(t0, tf) or channel in channels_to_plot or plot_all:
                waveform = events.get_waveform()
                v_max = max(v_max,
                            max(np.abs(np.real(waveform))),
                            max(np.abs(np.imag(waveform))))
                n_valid_waveform += 1
                events.enable = True

        # create table
        table_data = []
        if self.style.use_table:
            for channel, events in channels.items():
                table_data.extend(events.to_table(channel.name, (t0, tf)))
            table_data = sorted(table_data, key=lambda x: x[0])

        # plot table
        if len(table_data) > 0:
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
            cell_color = [[self.style.table_color * ncols for _jj in range(nrows)]]
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
                time, re, im = interp_method(events.get_waveform(), dt, self.style.num_points)
                # choose color
                if isinstance(channels, DriveChannel):
                    color = self.style.d_ch_color
                elif isinstance(channels, ControlChannel):
                    color = self.style.u_ch_color
                elif isinstance(channels, MeasureChannel):
                    color = self.style.m_ch_color
                else:
                    raise exceptions.VisualizationError('Ch %s cannot be drawn.' % channel.name)
                # scaling and offset
                if scaling:
                    v_max = 0.5 * scaling
                else:
                    v_max = 0.5 * 1.2 * v_max
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
                ax.plot((t0, tf), (0, 0), color='#000000', linewidth=1.0)

                # plot frame changes
                fcs = events.get_framechange(t0, tf)
                if len(fcs) > 0:
                    for time, fc in fcs.items():
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
