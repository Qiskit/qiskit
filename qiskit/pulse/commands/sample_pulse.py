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
from qiskit.pulse.timeslots import Interval, Timeslot, TimeslotCollection
from qiskit.pulse.exceptions import PulseError
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

    def draw(self, dt: float = 1, style=None,
             filename: str = None, interp_method: Callable = None,
             scaling: float = None, interactive: bool = False):
        """Plot the interpolated envelope of pulse.

        Args:
            dt: Time interval of samples.
            style (OPStylePulse): A style sheet to configure plot appearance.
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

    def __call__(self, channel: OutputChannel, name=None) -> 'DriveInstruction':
        if name is None:
            name = self.name
        return DriveInstruction(self, channel, name=name)


class DriveInstruction(Instruction):
    """Instruction to drive a pulse to an `OutputChannel`. """

    def __init__(self, command: SamplePulse, channel: OutputChannel, name=None):
        slots = [Timeslot(Interval(0, command.duration), channel)]
        super().__init__(command, TimeslotCollection(*slots), name=name)

    def __repr__(self):
        return '%s -> %s' % (self.command, self.channels[0])
