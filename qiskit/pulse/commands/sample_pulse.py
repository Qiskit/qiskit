# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Sample pulse.
"""
from typing import Set

from qiskit.pulse.channels import PulseChannel, OutputChannel
from qiskit.pulse.common.interfaces import Pulse
from qiskit.pulse.common.timeslots import Interval, Timeslot, TimeslotOccupancy
from .pulse_command import PulseCommand


class SamplePulse(PulseCommand):
    """Container for functional pulse."""

    def __init__(self, samples, name=None):
        """Create new sample pulse command.

        Args:
            samples (ndarray): Complex array of pulse envelope.
            name (str): Unique name to identify the pulse.
        """
        if not name:
            _name = str('pulse_object_%s' % id(self))
        else:
            _name = name

        super().__init__(duration=len(samples), name=_name)

        self.samples = samples

    def draw(self, **kwargs):
        """Plot the interpolated envelope of pulse.

        Keyword Args:
            dt (float): Time interval of samples.
            interp_method (str): Method of interpolation
                (set `None` for turn off the interpolation).
            filename (str): Name required to save pulse image.
            interactive (bool): When set true show the circuit in a new window
                (this depends on the matplotlib backend being used supporting this).
            dpi (int): Resolution of saved image.
            nop (int): Data points for interpolation.
            size (tuple): Size of figure.
        """
        from qiskit.tools.visualization import pulse_drawer

        return pulse_drawer(self.samples, self.duration, **kwargs)

    def __eq__(self, other):
        """Two SamplePulses are the same if they are of the same type
        and have the same name and samples.

        Args:
            other (SamplePulse): other SamplePulse

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self.name == other.name and \
                (self.samples == other.samples).all():
            return True
        return False

    def __call__(self, channel: OutputChannel) -> 'DrivePulse':
        return DrivePulse(self, channel)


class DrivePulse(Pulse):
    """Pulse to drive a pulse shape to a `OutputChannel`. """

    def __init__(self, command: SamplePulse, channel: OutputChannel):
        self._command = command
        self._channel = channel
        self._occupancy = TimeslotOccupancy([Timeslot(Interval(0, command.duration), channel)])

    @property
    def duration(self):
        return self._command.duration

    @property
    def channelset(self) -> Set[PulseChannel]:
        return {self._channel}

    @property
    def occupancy(self):
        return self._occupancy

    def __repr__(self):
        return '%s(name=%s, duration=%d, channelset=%s)' % \
               (self.__class__.__name__, self._command.name, self.duration, self.channelset)
