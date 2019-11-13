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
from .sample_pulse import SamplePulse
from .instruction import Instruction
from qiskit.pulse.pulse_lib import gaussian, gaussian_square, drag, constant


class ParametricPulse(Command):
    """
    """

    prefix = 'p'

    def __init__(self, duration):
        """
        """
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
        sampled_version = self.get_sample_pulse()
        return sampled_version.draw(dt=dt, style=style, filename=filename,
                                    interp_method=interp_method, scaling=scaling,
                                    interactive=interactive)

    def get_sample_pulse(self) -> SamplePulse:
        """
        """
        return NotImplementedError


    def to_instruction(self, channel: PulseChannel,
                       name: Optional[str] = None) -> 'ParametricInstruction':
        return ParametricInstruction(self, channel, name=name)


class Gaussian(ParametricPulse):
    """"""

    def __init__(self,
                 sigma: float,
                 amp: complex,
                 duration: int):
        """
        """
        self.params = {'sigma': sigma, 'amp': amp, 'duration': duration}
        self.sigma = sigma
        self.amp = amp
        super().__init__(duration=duration)

    def get_sample_pulse(self) -> SamplePulse:
        return gaussian(duration=self.duration, amp=self.amp,
                        sigma=self.sigma)

    def __repr__(self):
        return '{}(name={}, params={})'.format(self.__class__.__name__, self.name, self.params)


class GaussianSquare(ParametricPulse):
    """"""

    def __init__(self,
                 sigma: float,
                 amp: complex,
                 width: int,
                 duration: int):
        """
        """
        # args order dont match
        self.params = {'sigma': sigma, 'amp': amp, 'width': width, 'duration': duration}
        self.sigma = sigma
        self.amp = amp
        self.width = width
        super().__init__(duration=duration)

    def get_sample_pulse(self) -> SamplePulse:
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
        """
        """
        self.params = {
            'mean': mean,
            'sigma': sigma,
            'amp': amp,
            'beta': beta,
            'duration': duration}
        self.mean = mean
        self.sigma = sigma
        self.amp = amp
        self.beta = beta
        self.remove_baseline = remove_baseline
        super().__init__(duration=duration)

    def get_sample_pulse(self) -> SamplePulse:
        return drag(duration=self.duration, amp=self.amp, center=self.mean,
                    sigma=self.sigma, beta=self.beta, zero_ends=self.remove_baseline)


class SquarePulse(ParametricPulse):
    """"""

    def __init__(self,
                 value: complex,
                 duration: int):
        """
        """
        self.value = value
        super().__init__(duration=duration)

    def get_sample_pulse(self) -> SamplePulse:
        return constant(duration=self.duration, amp=self.value)


class ParametricInstruction(Instruction):
    """Instruction to drive a parametric pulse to an `PulseChannel`."""

    def __init__(self, command: ParametricPulse,
                 channel: PulseChannel,
                 name: Optional[str] = None):
        """
        """
        super().__init__(command, channel, name=name)

    def draw(self, **kwargs):
        """Plot the instruction.

        Args:
            dt: Time interval of samples
            style: A style sheet to configure plot appearance
            filename: Name required to save pulse image
            interp_method: A function for interpolation
            scaling: Relative visual scaling of waveform amplitudes
            channels_to_plot: A list of channel names to plot
            plot_all: Plot empty channels
            plot_range: A tuple of time range to plot
            interactive: When set true show the circuit in a new window
                (this depends on the matplotlib backend being used supporting this)
            table: Draw event table for supported commands
            label: Label individual instructions
            framechange: Add framechange indicators


        Returns:
            matplotlib.figure: A matplotlib figure object of the pulse schedule
        """
        return self.command.get_sample_pulse().draw(**kwargs)