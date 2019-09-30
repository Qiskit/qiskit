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

"""
Parametric pulses. These are pulses which are described by a known formula and some
parameters.
"""
from typing import Callable, Union, List, Optional

import numpy as np

from qiskit.pulse.channels import PulseChannel

from .command import Command
from .sample_pulse import SamplePulse, PulseInstruction
from qiskit.pulse.pulse_lib import gaussian, gaussian_square, drag, constant


class ParametricPulse(Command):
    """"""

    prefix = 'p'

    def __init__(self, duration):
        super().__init__(duration=duration)

    def draw(self, dt: float = 1,
             style: Optional['PulseStyle'] = None,
             filename: Optional[str] = None,
             interp_method: Optional[Callable] = None,
             scaling: float = 1, interactive: bool = False):
        """Plot the interpolated envelope of pulse.

        Args:
            dt: Time interval of samples.
            style: A style sheet to configure plot appearance
            filename: Name required to save pulse image
            interp_method: A function for interpolation
            scaling: Relative visual scaling of waveform amplitudes
            interactive: When set true show the circuit in a new window
                (this depends on the matplotlib backend being used supporting this)

        Returns:
            matplotlib.figure: A matplotlib figure object of the pulse envelope
        """
        # pylint: disable=invalid-name, cyclic-import
        sampled_version = self.get_samples()
        return sampled_version.draw(dt=dt, style=style, filename=filename,
                                    interp_method=interp_method, scaling=scaling,
                                    interactive=interactive)

    def get_samples(self):
        """
        """
        return NotImplementedError

    def __repr__(self):
        return '%s(%s, duration=%d)' % (self.__class__.__name__, self.name, self.duration)

    # pylint: disable=arguments-differ
    def to_instruction(self, channel: PulseChannel,
                       name: Optional[str] = None) -> 'PulseInstruction':
        return PulseInstruction(self, channel, name=name)
    # pylint: enable=arguments-differ


class Gaussian(ParametricPulse):
    """"""

    def __init__(self,
                 sigma: float,
                 amp: complex,
                 duration: int):
        self.sigma = sigma
        self.amp = amp
        super().__init__(duration=duration)

    def get_samples(self):
        return gaussian(duration=self.duration, amp=self.amp,
                        sigma=self.sigma)


class GaussianSquare(ParametricPulse):
    """"""

    def __init__(self,
                 sigma: float,
                 amp: complex,
                 width: int,
                 duration: int):
        self.sigma = sigma
        self.amp = amp
        self.width = width
        super().__init__(duration=duration)

    def get_samples(self):
        return gaussian_square(duration=self.duration, amp=self.amp,
                               width=self.width, sigma=self.sigma)


class Drag(ParametricPulse):
    """"""

    def __init__(self,
                 mean: float,
                 sigma: float,
                 amp: complex,
                 beta: int,
                 duration: int,
                 remove_baseline: bool = False):
        self.mean = mean
        self.sigma = sigma
        self.amp = amp
        self.beta = beta
        self.remove_baseline = remove_baseline
        super().__init__(duration=duration)

    def get_samples(self):
        return drag(duration=self.duration, amp=self.amp, center=self.mean,
                    sigma=self.sigma, beta=self.beta)


class SquarePulse(ParametricPulse):
    """"""

    def __init__(self,
                 value: complex,
                 duration: int):
        self.value = value
        super().__init__(duration=duration)

    def get_samples(self):
        return constant(duration=self.duration, amp=self.value)