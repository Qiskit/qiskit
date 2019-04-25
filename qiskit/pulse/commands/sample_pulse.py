# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Sample pulse.
"""
from typing import Callable

import numpy as np

from qiskit.pulse.channels import OutputChannel
from qiskit.pulse.common.timeslots import Interval, Timeslot, TimeslotCollection
from qiskit.pulse.exceptions import PulseError
from qiskit.visualization.qcstyle import OPStylePulse
from .instruction import Instruction
from .pulse_command import PulseCommand


class SamplePulse(PulseCommand):
    """Container for functional pulse."""

    def __init__(self, samples, name=None):
        """Create new sample pulse command.

        Args:
            samples (ndarray): Complex array of pulse envelope.
            name (str): Unique name to identify the pulse.
        Raises:
            PulseError: when pulse envelope amplitude exceeds 1.
        """
        super().__init__(duration=len(samples), name=name)

        if np.any(np.abs(samples) > 1):
            raise PulseError('Absolute value of pulse envelope amplitude exceeds 1.')

        self._samples = samples

    @property
    def samples(self):
        """Return sample values."""
        return self._samples

    def draw(self, dt: float = 1, style: OPStylePulse = None,
             filename: str = None, interp_method: Callable = None,
             scaling: float = None, interactive: bool = False):
        """Plot the interpolated envelope of pulse.

        Args:
            dt: Time interval of samples.
            style: A style sheet to configure plot appearance.
            filename: Name required to save pulse image.
            interp_method: A function for interpolation.
            scaling: scaling of waveform amplitude.
            interactive: When set true show the circuit in a new window
                (this depends on the matplotlib backend being used supporting this).

        Returns:
            matplotlib.figure: A matplotlib figure object of the pulse envelope.
        """
        # pylint: disable=invalid-name, cyclic-import

        from qiskit.tools import visualization

        return visualization.pulse_drawer(self, dt=dt, style=style, filename=filename,
                                          interp_method=interp_method, scaling=scaling,
                                          interactive=interactive)

    def __eq__(self, other):
        """Two SamplePulses are the same if they are of the same type
        and have the same name and samples.

        Args:
            other (SamplePulse): other SamplePulse

        Returns:
            bool: are self and other equal.
        """
        if super().__eq__(other) and \
                (self._samples == other._samples).all():
            return True
        return False

    def __hash__(self):
        return hash((super().__hash__(), self._samples.tostring()))

    def __repr__(self):
        return '%s(%s, duration=%d)' % (self.__class__.__name__, self.name, self.duration)

    def __call__(self, channel: OutputChannel) -> 'DriveInstruction':
        return DriveInstruction(self, channel)


class DriveInstruction(Instruction):
    """Instruction to drive a pulse to an `OutputChannel`. """

    def __init__(self, command: SamplePulse, channel: OutputChannel, start_time: int = 0):
        slots = [Timeslot(Interval(start_time, start_time + command.duration), channel)]
        super().__init__(command, start_time, TimeslotCollection(slots))
        self._channel = channel

    @property
    def command(self) -> SamplePulse:
        """SamplePulse command. """
        return self._command

    @property
    def channel(self) -> OutputChannel:
        """OutputChannel command. """
        return self._channel

    def __repr__(self):
        return '%4d: %s -> %s' % (self._start_time, self._command, self._channel)
