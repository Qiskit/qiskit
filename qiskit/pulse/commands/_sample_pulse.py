# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Sample pulse.
"""

from qiskit.pulse.commands import PulseCommand


class SamplePulse(PulseCommand):
    """Container for functional pulse."""

    def __init__(self, duration, samples, name):
        """Create new sample pulse command.

        Args:
            duration (int): Duration of pulse.
            samples (ndarray): Complex array of pulse envelope.
            name (str): Unique name to identify the pulse.
        """

        super(SamplePulse, self).__init__(duration=duration, name=name)

        self._samples = samples

    @property
    def samples(self):
        """Return sample.
        """
        return self._samples

    @samples.setter
    def samples(self, samples):
        """Set sample.
        """
        self._samples = samples
        self.duration = len(samples)

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
        from qiskit.tools import visualization

        return visualization.pulse_drawer(self.samples, self.duration, **kwargs)
