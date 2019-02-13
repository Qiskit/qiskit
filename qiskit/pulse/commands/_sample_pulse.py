# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Sample pulse.
"""

from qiskit.pulse.commands._pulse_command import PulseCommand


class SamplePulse(PulseCommand):
    """Container for functional pulse."""

    def __init__(self, duration):
        """create new sample pulse command.

        Args:
            duration (int): duration of pulse
        """

        super(SamplePulse, self).__init__(duration)

        self.__sample = []

    @property
    def sample(self):
        """ Return sample
        """
        return self.__sample

    @sample.setter
    def sample(self, sample):
        """ Set sample
        """
        self.__sample = sample
        self.duration = len(sample)

    def draw(self, **kwargs):
        """Plot the interpolated envelope of pulse

        Keyword Args:
            dt (float): time interval of samples
            interp_method (str): method of interpolation
                (set `None` for turn off the interpolation)
            filename (str): name required to save pulse image
            interactive (bool): when set true show the circuit in a new window
                (for `mpl` this depends on the matplotlib backend being used
                supporting this).
            dpi (int): resolution of saved image
            nop (int): data points for interpolation
            size (tuple): size of figure
        """
        from qiskit.tools import visualization

        return visualization.pulse_drawer(self.sample, self.duration, **kwargs)
