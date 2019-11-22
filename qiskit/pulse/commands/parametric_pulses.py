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

If a backend supports parametric pulses, it will have the attribute
`backend.configuration().parametric_pulses`, which is a list of supported pulse shapes, such as
`['gaussian', 'gaussian_square', 'drag']`. A Pulse Schedule, using parametric pulses, which is
assembled for a backend which supports those pulses, will result in a Qobj which is dramatically
smaller than one which uses SamplePulses.

This module can easily be extended to describe more pulse shapes. The new class should:
  - have a descriptive name
  - be a well known and/or well described formula (include the formula in the class docstring)
  - take some parameters (at least `duration`)
  - implement a `get_sample_pulse` method which returns a corresponding SamplePulse in the
    case that it is assembled for a backend which does not support it.

The new pulse must then be registered by the assembler in
`qiskit/qobj/converters/pulse_instruction.py:ParametricPulseShapes`
by following the existing pattern:
    class ParametricPulseShapes(Enum):
        gaussian = commands.Gaussian
        ...
        new_supported_pulse_name = commands.YourPulseCommandClass
"""
from abc import abstractmethod
from typing import Any, Dict, Optional

from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.pulse_lib.discrete import gaussian, gaussian_square, drag, constant

from .command import Command
from .sample_pulse import SamplePulse
from .instruction import Instruction

# pylint: disable=missing-docstring


class ParametricPulse(Command):
    """The abstract superclass for parametric pulses."""
    prefix = 'p'

    @abstractmethod
    def __init__(self, duration: int):
        """Create a parametric pulse command.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
        """
        super().__init__(duration=duration)

    @abstractmethod
    def get_sample_pulse(self) -> SamplePulse:
        """Return a SamplePulse with samples filled according to the formula that the pulse
        represents and the parameter values it contains.
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """Return a dictionary containing the parameters specific to this parametric pulse."""
        return {attr[1:]: getattr(self, attr)
                for attr in self.__dict__
                if attr.startswith('_') and attr != '_name'}

    def to_instruction(self, channel: PulseChannel,
                       name: Optional[str] = None) -> 'ParametricInstruction':
        # pylint: disable=arguments-differ
        return ParametricInstruction(self, channel, name=name)

    def __repr__(self):
        return '{}(name={}, params={})'.format(self.__class__.__name__,
                                               self.name,
                                               self.get_params())


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
        validate_params(amp=amp, sigma=sigma)
        self._amp = complex(amp)
        self._sigma = sigma
        super().__init__(duration=duration)

    @property
    def amp(self):
        return self._amp

    @property
    def sigma(self):
        return self._sigma

    def get_sample_pulse(self) -> SamplePulse:
        return gaussian(duration=self.duration, amp=self.amp,
                        sigma=self.sigma, zero_ends=False)


class GaussianSquare(ParametricPulse):
    """
    A square pulse with a Gaussian shaped risefall on either side:

        risefall = (duration - width) / 2

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
                 width: float):
        """Initialize the gaussian square command.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the Gaussian and of the square pulse.
            sigma: A measure of how wide or narrow the Gaussian risefall is; described
                   mathematically in the class docstring.
            width: The duration of the embedded square pulse.
        """
        validate_params(amp=amp, sigma=sigma, width=width, duration=duration)
        self._amp = complex(amp)
        self._sigma = sigma
        self._width = width
        super().__init__(duration=duration)

    @property
    def amp(self):
        return self._amp

    @property
    def sigma(self):
        return self._sigma

    @property
    def width(self):
        return self._width

    def get_sample_pulse(self) -> SamplePulse:
        return gaussian_square(duration=self.duration, amp=self.amp,
                               width=self.width, sigma=self.sigma,
                               zero_ends=False)


class Drag(ParametricPulse):
    """
    A pulse whose envelope is shaped by a drag pulse. This is so named by the technique
    Derivative Removal by Adiabatic Gate (DRAG). It is constructed to reduce the frequency
    spectrum of a normal gaussian curve near the |1>-|2> transition.

        f(x) = Gaussian + 1j * beta * d/dx [Gaussian]
             = Gaussian + 1j * beta * (-x / sigma) [Gaussian]

    where 'Gaussian' is:
        Gaussian(x, amp, sigma) = amp * exp( -(1/2) * (x - duration/2)^2 / sigma^2) )

    Ref:
        [1] Gambetta, J. M., Motzoi, F., Merkel, S. T. & Wilhelm, F. K.
        Analytic control methods for high-fidelity unitary operations
        in a weakly nonlinear oscillator. Phys. Rev. A 83, 012308 (2011).
        [2] F. Motzoi, J. M. Gambetta, P. Rebentrost, and F. K. Wilhelm
        Phys. Rev. Lett. 103, 110501 – Published 8 September 2009
    """

    def __init__(self,
                 duration: int,
                 amp: complex,
                 sigma: float,
                 beta: int):
        """Initialize the drag command.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the Drag envelope.
            sigma: A measure of how wide or narrow the Gaussian peak is; described mathematically
                   in the class docstring.
            beta: The correction amplitude.
        """
        validate_params(amp=amp, sigma=sigma, beta=beta)
        self._amp = complex(amp)
        self._sigma = sigma
        self._beta = beta
        super().__init__(duration=duration)

    @property
    def amp(self):
        return self._amp

    @property
    def sigma(self):
        return self._sigma

    @property
    def beta(self):
        return self._beta

    def get_sample_pulse(self) -> SamplePulse:
        return drag(duration=self.duration, amp=self.amp, sigma=self.sigma,
                    beta=self.beta, zero_ends=False)


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
        validate_params(amp=amp)
        self._amp = complex(amp)
        super().__init__(duration=duration)

    @property
    def amp(self):
        return self._amp

    def get_sample_pulse(self) -> SamplePulse:
        return constant(duration=self.duration, amp=self.amp)


def validate_params(**kwargs):
    """
    Raise a PulseError if the parameters passed are not valid.
    """
    if 'amp' in kwargs:
        if abs(kwargs['amp']) > 1.:
            raise PulseError("The amplitude norm must be less than 1, "
                             "found: {}".format(abs(kwargs['amp'])))
    if 'sigma' in kwargs:
        if kwargs['sigma'] <= 0:
            raise PulseError("Sigma must be greater than 0.")
    if 'beta' in kwargs:
        if isinstance(kwargs['beta'], complex):
            raise PulseError("Beta must be real.")
    if 'width' in kwargs:
        if kwargs['width'] < 0:
            raise PulseError("Width cannot be less than 0.")
        if 'duration' in kwargs and kwargs['width'] >= kwargs['duration']:
            raise PulseError("The width of the pulse must be less than its duration.")


class ParametricInstruction(Instruction):
    """Instruction to drive a parametric pulse to an `PulseChannel`."""
