# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
mpl pulse visualization.
"""

import numpy as np
import logging
from scipy.interpolate import CubicSpline
from collections import OrderedDict

from qiskit.pulse.schedule import TimedPulse, PulseSchedule
from qiskit.pulse.commands import SamplePulse, FunctionalPulse, FrameChange, PersistentValue

from qiskit.tools.visualization import exceptions


try:
    from matplotlib import pyplot as plt, gridspec

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


def pulse_drawer(data, dt=1, interp_method='None',
                 style=None, filename=None,
                 interactive=False, plot_channels=None,
                 plot_empty=False, plot_range=None):
    """Plot the interpolated envelope of pulse

    Args:
        data (PulseSchedule or SamplePulse): Data to plot.
        dt (float): Time interval of samples.
        interp_method (str): A method of interpolation.
            'None' for turn off interpolation
            'CubicSpline' for cubic spline interpolation
        filename (str): Name required to save pulse image.
        style (dict): A style sheet to configure plot appearance.
        interactive (bool): When set true show the circuit in a new window
            (this depends on the matplotlib backend being used supporting this).
        plot_empty (bool): Plot empty channels.
        plot_channels (list): A list of channel names to plot.
        plot_range (tuple): A tuple of time range to plot.
    Returns:
        matplotlib.figure: A matplotlib figure object for the pulse envelope.
    Raises:
        QiskitError: when invalid data is given.
    """
    _style = op_default()
    if style:
        _style.update(style)

    drawer = PulseDrawer(_style)

    if isinstance(data, (SamplePulse, FunctionalPulse)):
        image = drawer.draw_sample(data, dt, interp_method)
    elif isinstance(data, PulseSchedule):
        image = drawer.draw_schedule(data, dt, interp_method,
                                     plot_empty, plot_channels, plot_range)
    else:
        raise exceptions.VisualizationError('This data cannot be visualized.')

    if filename:
        image.savefig(filename, dpi=_style['dpi'], bbox_inches='tight')

    plt.close(image)

    if image and interactive:
        image.show()
    return image


def op_default():
    """Pulse default style.
    """
    return {
        'sched2d':
            {
                'colors': {
                    'd': ['#648fff', '#002999'],
                    'u': ['#ffb000', '#994A00'],
                    'm': ['#dc267f', '#760019'],
                    'table': ['#e0e0e0', '#f6f6f6', '#f6f6f6']
                },
                'fig_w': 10,
                'use_table': True,
                'table_font': 10,
                'label_font': 18,
                'table_cols': 2,
                'table_row_height': 0.4,
                'pulse_row_height': 2.5
            },
        'sample':
            {
                'color': {
                    'real': '#ff0000',
                    'imag': '#0000ff'
                },
                'fig_w': 6,
                'fig_h': 5
            },
        'bg_color': '#f2f3f4',
        'num_points': 1000,
        'dpi': 150
    }


class Channels:
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

        self.enable = False

    def add_timedpulse(self, pulse):
        """Add new timed pulse to channel.

        Args:
            pulse (TimedPulse): TimePulse object to be added.
        """
        command = pulse.command

        if isinstance(command, (SamplePulse, FunctionalPulse)):
            self.samples[pulse.t0:pulse.t0+command.duration] += command.samples
        elif isinstance(command, FrameChange):
            if pulse.t0 in self.fc_pulses.keys():
                self.fc_pulses[pulse.t0] += command.phase
            else:
                self.fc_pulses[pulse.t0] = command.phase
        elif isinstance(command, PersistentValue):
            if pulse.t0 in self.pv_pulses.keys():
                self.pv_pulses[pulse.t0] += command.value
            else:
                self.pv_pulses[pulse.t0] = command.value
        else:
            pass

    def get_waveform(self):
        """Get waveform.
        """
        # sort fc by time index
        fcs = sorted(self.fc_pulses.items(), key=lambda x: x[0])
        fc_t = np.ones_like(self.samples)
        for t0, val in fcs:
            fc_t[t0:] *= np.exp(1j * val)
        pvs = sorted(self.pv_pulses.items(), key=lambda x: x[0])
        # sort pv by time index
        pv_t = np.zeros_like(self.samples)
        for t0, val in pvs:
            pv_t[t0:] = val

        return fc_t * (self.samples + pv_t)

    def is_empty(self, time_range=None):
        """Return if pulse is empty.
        """
        waveform = self.get_waveform()

        if time_range:
            t0, tf = time_range
            _fc_times = np.array(list(self.fc_pulses.keys()))
            if any(waveform[t0:tf + 1]):
                return False
            elif any(_fc_times) and any(t0 <= _fc_times <= tf):
                return False
            else:
                return True
        else:
            if any(waveform) or len(self.fc_pulses) > 0:
                return False
            else:
                return True

    def to_table(self, name):
        """Get table contains.

        Args:
            name: name of channel.
        """
        time_event = []

        for key, val in self.fc_pulses.items():
            data_str = 'FrameChange, %.2f' % val
            time_event.append((key, name, data_str))
        for key, val in self.conditionals.items():
            data_str = 'Conditional, %s' % val
            time_event.append((key, name, data_str))

        return time_event


class PulseDrawer:
    def __init__(self, style):
        """Create new figure.

        Args:
            style (dict): A style sheet to configure plot appearance.
        """
        self.style = style

        self.figure = plt.figure()

    def draw_schedule(self, schedule, dt, interp_method,
                      plot_empty=False, plot_channels=None,
                      plot_range=None):
        """Draw pulse schedules.

        Args:
            schedule (PulseSchedule): PulseSchedule to draw.
            dt (float): Time interval of samples.
            interp_method (str): A method of interpolation.
            plot_empty (bool): Plot empty channels.
            plot_channels (list): A list of channel names to plot.
            plot_range (tuple): A tuple of time range to plot.
        """
        if plot_range:
            t0 = int(np.floor(plot_range[0]/dt))
            tf = int(np.floor(plot_range[1]/dt))
            _times = t0, tf
        else:
            _times = None

        # generate channels
        regs = [
            schedule.channels.drive,
            schedule.channels.control,
            schedule.channels.measure
        ]
        chs_dict = OrderedDict()
        tf = schedule.end_time()
        for chs in zip(*regs):
            for ch in chs:
                chs_dict[ch.name] = Channels(tf)

        # add pulses
        for pulse in schedule.flat_pulse_sequence():
            chs_dict[pulse.channel.name].add_timedpulse(pulse)

        # check active channels
        table_data = []
        n_channels = 0
        for name, channel in chs_dict.items():
            table_data.extend(channel.to_table(name))
            if not channel.is_empty(time_range=_times) or plot_empty:
                if plot_channels:
                    if name in plot_channels:
                        channel.enable = True
                        n_channels += 1
                else:
                    channel.enable = True
                    n_channels += 1
        table_data = sorted(table_data, key=lambda x: x[0])

        # plot table
        default = self.style['sched2d']
        if default['use_table']:
            # height
            ncols = default['table_cols']
            nrows = int(np.ceil(len(table_data)/ncols))
            _th = nrows * default['table_row_height']
            _ah = n_channels * default['pulse_row_height']
            fig_h = _th + _ah
            # object
            gs = gridspec.GridSpec(2, 1, height_ratios=[_th, _ah], hspace=0)
            tb = plt.subplot(gs[0])
            tb.axis('off')
            # generate table
            _table_value = [
                ['' for _kk in range(ncols*3)]
                for _jj in range(nrows)
            ]
            _table_color = [
                default['colors']['table'] * ncols
                for _jj in range(nrows)
            ]
            _col_width = [
                *([0.2, 0.2, 0.5] * ncols)
            ]
            for ii, item in enumerate(table_data):
                _r, _c = np.unravel_index(ii, (nrows, ncols), order='f')
                _t, _ch, _dstr = item
                # time
                _table_value[_r][3*_c+0] = 't = %s' % _t * dt
                # channel
                _table_value[_r][3*_c+1] = 'ch %s' % _ch
                # description
                _table_value[_r][3*_c+2] = _dstr
            _table = tb.table(cellText=_table_value,
                              cellLoc='left',
                              rowLoc='center',
                              colWidths=_col_width,
                              bbox=[0, 0, 1, 1],
                              cellColours=_table_color)
            _table.auto_set_font_size(False)
            _table.set_fontsize = default['table_font']
            ax = plt.subplot(gs[1])
        else:
            # height
            fig_h = n_channels * default['pulse_row_height']
            # object
            ax = self.figure.add_subplot(111)
        self.figure.set_size_inches(default['fig_w'], fig_h)
        ax.set_facecolor(self.style['bg_color'])

        # plot waveforms
        colors = self.style['sched2d']['colors']

        y0 = 0
        for name, channel in chs_dict.items():
            if channel.enable:
                # plot waveform
                time, re, im = self.interp(channel.get_waveform(), dt, self.style['num_points'],
                                           interp_method)
                re = 0.5 * re + y0
                im = 0.5 * im + y0
                offset = np.zeros_like(time) + y0
                ax.fill_between(x=time, y1=re, y2=offset,
                                facecolor=colors[name[0]][0], alpha=0.3,
                                edgecolor=colors[name[0]][0], linewidth=1.5,
                                label='real part')
                ax.fill_between(x=time, y1=im, y2=offset,
                                facecolor=colors[name[0]][1], alpha=0.3,
                                edgecolor=colors[name[0]][1], linewidth=1.5,
                                label='imaginary part')
                ax.plot((time[0], time[1]), (0, 0), color='#000000', linewidth=1.0)
                # plot fcs
                if len(channel.fc_pulses) > 0:
                    for time, val in channel.fc_pulses.items():
                        if plot_range:
                            if time > plot_range[1] or time < plot_range[0]:
                                continue
                        ax.text(x=time*dt, y=y0, s=r'$\circlearrowleft$',
                                fontsize=default['label_font'], ha='center', va='center')
                # plot label
                ax.text(x=0, y=y0, s=name, fontsize=default['label_font'],
                        ha='right', va='center')
            else:
                continue
            y0 -= 1

        if plot_range:
            ax.set_xlim(plot_range)
        else:
            ax.set_xlim(0, schedule.end_time() * dt)
        ax.set_ylim(y0, 1)
        ax.set_yticklabels([])

        return self.figure

    def draw_sample(self, pulse, dt, interp_method):
        """Draw sample pulses.

        Args:
            pulse (SamplePulse): SamplePulse to draw.
            dt (float): Time interval of samples.
            interp_method (str): A method of interpolation.
        """
        self.figure.set_size_inches(self.style['sample']['fig_w'],
                                    self.style['sample']['fig_h'])
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(self.style['bg_color'])

        samples = pulse.samples
        time, re, im = self.interp(samples, dt, self.style['num_points'], interp_method)

        # plot
        colors = self.style['sample']['color']

        ax.fill_between(x=time, y1=re, y2=np.zeros_like(time),
                        facecolor=colors['real'], alpha=0.3,
                        edgecolor=colors['real'], linewidth=1.5,
                        label='real part')
        ax.fill_between(x=time, y1=im, y2=np.zeros_like(time),
                        facecolor=colors['imag'], alpha=0.3,
                        edgecolor=colors['imag'], linewidth=1.5,
                        label='imaginary part')

        ax.set_xlim(0, len(samples) * dt)
        ax.grid(b=True, linestyle='-')
        ax.legend(bbox_to_anchor=(0.5, 1.00), loc='lower center',
                  ncol=2, frameon=False, fontsize=14)

        return self.figure

    @staticmethod
    def interp(samples, dt, nop, interp_method):
        """Interpolate interval time.

        Args:
            samples (ndarray): A list of complex pulse envelope.
            dt (float): Time interval of samples.
            nop (int): Data points for interpolation.
            interp_method (str): A method of interpolation.

        Returns:
            tuple: Timebase, real and imaginary part of pulse envelope.

        Raises:
            VisualizationError: when invalid interp method is specified.
        """
        re_y = np.real(samples)
        im_y = np.imag(samples)

        if interp_method == 'CubicSpline':
            # spline interpolation, use mid-point of dt
            time = (np.arange(0, len(samples) + 1) + 0.5) * dt
            cs_ry = CubicSpline(time[:-1], re_y)
            cs_iy = CubicSpline(time[:-1], im_y)

            _time = np.linspace(0, len(samples) * dt, nop)
            _re_y = cs_ry(_time)
            _im_y = cs_iy(_time)
        elif interp_method == 'None':
            # pseudo-DAC output
            time = np.arange(0, len(samples) + 1) * dt

            _time = np.r_[time[0], np.repeat(time[1:-1], 2), time[-1]]
            _re_y = np.repeat(re_y, 2)
            _im_y = np.repeat(im_y, 2)
        else:
            raise exceptions.VisualizationError('Invalid interpolation method.')

        return _time, _re_y, _im_y
