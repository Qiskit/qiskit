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

from qiskit.pulse.schedule import TimedPulse, PulseSchedule
from qiskit.pulse.commands import SamplePulse, FunctionalPulse, FrameChange, PersistentValue

from qiskit.tools.visualization import exceptions


try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


def pulse_drawer(data, dt=1, interp_method='None',
                 filename=None, interactive=False,
                 dpi=150, nop=1000, size=(6, 5)):
    """Plot the interpolated envelope of pulse

    Args:
        data (PulseSchedule or SamplePulse): Data to plot.
        dt (float): Time interval of samples.
        nop (int): Data points for interpolation.
        interp_method (str): A method of interpolation.
            'None' for turn off interpolation
            'CubicSpline' for cubic spline interpolation
        filename (str): Name required to save pulse image.
        interactive (bool): When set true show the circuit in a new window
            (this depends on the matplotlib backend being used supporting this).
        dpi (int): Resolution of saved image.
        size (tuple): Size of figure.
    Returns:
        matplotlib.figure: A matplotlib figure object for the pulse envelope.
    Raises:
        QiskitError: when invalid data is given.
    """

    drawer = PulseDrawer(size)

    if isinstance(data, (SamplePulse, FunctionalPulse)):
        image = drawer.draw_sample(data, dt, nop, interp_method)
    elif isinstance(data, PulseSchedule):
        image = drawer.draw_schedule(data, dt, nop, interp_method)
    else:
        raise exceptions.VisualizationError('This data cannot be visualized.')

    if filename:
        image.savefig(filename, dpi=dpi, bbox_inches='tight')

    plt.close(image)

    if image and interactive:
        image.show()
    return image


class Channels:
    """Pulse dataset for channel."""

    def __init__(self, duration):
        """Create new channel dataset.

        Args:
            duration (int):
        """

        self.sample = np.zeros(duration + 1)
        self.fc_pulses = {}
        self.pv_pulses = {}
        self.conditionals = {}
        self.graph_offset = 0

    def add_timedpulse(self, pulse):
        """Add new timed pulse to channel.

        Args:
            pulse (TimedPulse): TimePulse object to be added.
        """
        command = pulse.command

        if isinstance(command, (SamplePulse, FunctionalPulse)):
            self.sample[pulse.t0:pulse.t0+command.duration] += command.samples
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

    def __call__(self):
        """Get waveform.
        """
        # sort fc by time index
        fcs = sorted(self.fc_pulses.items(), key=lambda x: x[0])
        fc_t = np.ones_like(self.sample)
        for t0, val in fcs:
            fc_t[t0:] *= np.exp(1j * val)
        pvs = sorted(self.pv_pulses.items(), key=lambda x: x[0])
        # sort pv by time index
        pv_t = np.zeros_like(self.sample)
        for t0, val in pvs:
            pv_t[t0:] = val

        return fc_t * (self.sample + pv_t)


class PulseDrawer:
    def __init__(self, size):
        """Create new figure.

        Args:
            size (tuple): figure size.
        """
        self.figure = plt.figure(figsize=size)
        self.ax = self.figure.add_subplot(111)

    def draw_schedule(self, schedule, dt, nop, interp_method,
                      plot_actives):
        """Draw pulse schedules.

        Args:
            schedule (PulseSchedule): PulseSchedule to draw.
            dt (float): Time interval of samples.
            nop (int): Data points for interpolation.
            interp_method (str): A method of interpolation.
            plot_actives (bool): Plot only pulse containing channels.
        """
        # generate channels
        regs = schedule.channel_bank.d, schedule.channel_bank.u, schedule.channel_bank.m
        chs_dict = {}
        for reg in regs:
            for ch in reg:
                chs_dict[ch.name] = Channels(schedule.end_time())

        # add pulses
        for pulse in schedule.flat_pulse_sequence():
            chs_dict[pulse.channel.name].add_timedpulse(pulse)

        # plot
        if plot_actives:


        return self.figure

    def draw_sample(self, pulse, dt, nop, interp_method):
        """Draw sample pulses.

        Args:
            pulse (SamplePulse): SamplePulse to draw.
            dt (float): Time interval of samples.
            nop (int): Data points for interpolation.
            interp_method (str): A method of interpolation.
        """
        samples = pulse.samples
        time, re, im = self.interp(samples, dt, nop, interp_method)

        # plot
        self.ax.fill_between(x=time, y1=re, y2=np.zeros_like(time),
                             facecolor='red', alpha=0.3,
                             edgecolor='red', linewidth=1.5,
                             label='real part')
        self.ax.fill_between(x=time, y1=im, y2=np.zeros_like(time),
                             facecolor='blue', alpha=0.3,
                             edgecolor='blue', linewidth=1.5,
                             label='imaginary part')

        self.ax.set_xlim(0, len(samples) * dt)
        self.ax.grid(b=True, linestyle='-')
        self.ax.legend(bbox_to_anchor=(0.5, 1.00), loc='lower center',
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
