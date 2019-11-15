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
Parametric pulse commands. These are pulse commands which are described by a known formula and some
parameters.

This module can easily be extended to describe more pulse shapes. The new class should:
  - have a descriptive name
  - be a well known and/or well described formula
  - take some parameters (at least `duration`)
  - implement a `get_sample_pulse` method to convert itself to a SamplePulse in the
    case that it is assembled for a backend which does not support it.
The new pulse shape should then be added to
`qiskit/qobj/converters/pulse_instruction.py:ParametricPulseShapes`.

The usefulness of these classes are limited by pulse backends supporting them. If a backend
supports parametric pulses, it will have the attribute
`backend.configuration().parametric_pulses`, which is a list of supported pulse shapes, such as
`['gaussian', 'gaussian_square', 'drag']`.
"""
from abc import abstractmethod
from typing import Optional

from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.pulse_lib.discrete import gaussian, gaussian_square, drag, constant

from .command import Command
from .sample_pulse import SamplePulse
from .instruction import Instruction


class ParametricPulse(Command):
    """The abstract superclass for parametric pulses."""
    prefix = 'p'

    @abstractmethod
    def __init__(self, duration: int):
        """Create a parametric pulse command.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
        """
        if not hasattr(self, 'params'):
            self.params = {'duration': duration}
        super().__init__(duration=duration)

    @abstractmethod
    def get_sample_pulse(self) -> SamplePulse:
        """Return a SamplePulse with samples filled according to the formula that the pulse
        represents and the parameter values it contains.
        """
        pass

    def to_instruction(self, channel: PulseChannel,
                       name: Optional[str] = None) -> 'ParametricInstruction':
        # pylint: disable=arguments-differ
        return ParametricInstruction(self, channel, name=name)

    def __repr__(self):
        return '{}(name={}, params={})'.format(self.__class__.__name__, self.name, self.params)


class Gaussian(ParametricPulse):
    """
    A truncated pulse envelope shaped according to the Gaussian function whose mean is centered at
    the center of the pulse (duration / 2):

        f(x) = amp * exp( -(1/2) * (x - duration/2)^2 / sigma^2) )  ,  0 <= x < duration
    """

    def __init__(self,
                 duration: int,
                 amp: complex,
                 sigma: float):
        """Initialize the gaussian command.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the Gaussian envelope.
            sigma: A measure of how wide or narrow the Gaussian peak is; described mathematically
                   in the class docstring.
        """
        self.params = {'sigma': sigma, 'amp': amp, 'duration': duration}  # FIXME
        self.sigma = sigma
        self.amp = amp
        super().__init__(duration=duration)

    def get_sample_pulse(self) -> SamplePulse:
        return gaussian(duration=self.duration, amp=self.amp,
                        sigma=self.sigma)


class GaussianSquare(ParametricPulse):
    """
    A square pulse with a Gaussian shaped risefall on either side:

        risefall = duration - width / 2

    0 <= x < risefall
        f(x) = amp * exp( -(1/2) * (x - risefall/2)^2 / sigma^2) )

    risefall <= x < risefall + width
        f(x) = amp

    risefall + width <= x < duration
        f(x) = amp * exp( -(1/2) * (x - (risefall + width)/2)^2 / sigma^2) )
    """

    def __init__(self,
                 duration: int,
                 amp: complex,
                 sigma: float,
                 width: int):
        """Initialize the gaussian square command.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the Gaussian and of the square pulse.
            sigma: A measure of how wide or narrow the Gaussian risefall is; described
                   mathematically in the class docstring.
            width: The duration of the embedded square pulse.
        """
        # args order dont match
        self.params = {'sigma': sigma, 'amp': amp, 'width': width, 'duration': duration}
        self.sigma = sigma
        self.amp = amp
        self.width = width
        super().__init__(duration=duration)

    def get_sample_pulse(self) -> SamplePulse:
        return gaussian_square(duration=self.duration, amp=self.amp,
                               risefall=(self.duration - self.width) / 2, sigma=self.sigma)


class Drag(ParametricPulse):
    """
    A pulse whose envelope is shaped by a drag pulse. This is so named by the technique
    Derivative Removal by Adiabatic Gate (DRAG).
    """

    def __init__(self,
                 duration: int,
                 amp: complex,
                 sigma: float,
                 beta: int,
                 remove_baseline: bool = False):
        """Initialize the drag command.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the Drag envelope.
            sigma:
            beta:
            remove_baseline:
        """
        self.params = {
            'sigma': sigma,
            'amp': amp,
            'beta': beta,
            'duration': duration}
        self.sigma = sigma
        self.amp = amp
        self.beta = beta
        self.remove_baseline = remove_baseline
        super().__init__(duration=duration)

    def get_sample_pulse(self) -> SamplePulse:
        return drag(duration=self.duration, amp=self.amp, sigma=self.sigma,
                    beta=self.beta, zero_ends=self.remove_baseline)


class ConstantPulse(ParametricPulse):
    """
    A simple constant pulse, with an amplitude value and a duration:

        f(x) = amp    ,  0 <= x < duration
        f(x) = 0      ,  elsewhere
    """

    def __init__(self,
                 duration: int,
                 amp: complex):
        """
        Initialize the constant valued pulse command.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the constant square pulse.
        """
        self.params = {'amp': amp, 'duration': duration}
        self.amp = amp
        super().__init__(duration=duration)

    def get_sample_pulse(self) -> SamplePulse:
        return constant(duration=self.duration, amp=self.amp)


class ParametricInstruction(Instruction):
    """Instruction to drive a parametric pulse to an `PulseChannel`."""
