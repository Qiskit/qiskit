# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-return-doc, invalid-name

"""Module for builtin discrete pulses.

Note the sampling strategy use for all discrete pulses is `midpoint`.
"""
from typing import Optional

from qiskit.pulse.pulse_lib import continuous
from qiskit.pulse.exceptions import PulseError

from . import samplers


_sampled_constant_pulse = samplers.midpoint(continuous.constant)


def constant(duration: int, amp: complex, name: Optional[str] = None) -> 'SamplePulse':
    """Generates constant-sampled `SamplePulse`.

    Applies `midpoint` sampling strategy to generate discrete pulse from continuous function.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Complex pulse amplitude.
        name: Name of pulse.
    """
    return _sampled_constant_pulse(duration, amp, name=name)


_sampled_zero_pulse = samplers.midpoint(continuous.zero)


def zero(duration: int, name: Optional[str] = None) -> 'SamplePulse':
    """Generates zero-sampled `SamplePulse`.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        name: Name of pulse.
    """
    return _sampled_zero_pulse(duration, name=name)


_sampled_square_pulse = samplers.midpoint(continuous.square)


def square(duration: int, amp: complex, period: float = None,
           phase: float = 0, name: Optional[str] = None) -> 'SamplePulse':
    """Generates square wave `SamplePulse`.

    Applies `midpoint` sampling strategy to generate discrete pulse from continuous function.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude. Wave range is [-amp, amp].
        period: Pulse period, units of dt. If `None` defaults to single cycle.
        phase: Pulse phase.
        name: Name of pulse.
    """
    if period is None:
        period = duration

    return _sampled_square_pulse(duration, amp, period, phase=phase, name=name)


_sampled_sawtooth_pulse = samplers.midpoint(continuous.sawtooth)


def sawtooth(duration: int, amp: complex, period: float = None,
             phase: float = 0, name: Optional[str] = None) -> 'SamplePulse':
    """Generates sawtooth wave `SamplePulse`.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude. Wave range is [-amp, amp].
        period: Pulse period, units of dt. If `None` defaults to single cycle.
        phase: Pulse phase.
        name: Name of pulse.
    """
    if period is None:
        period = duration

    return _sampled_sawtooth_pulse(duration, amp, period, phase=phase, name=name)


_sampled_triangle_pulse = samplers.midpoint(continuous.triangle)


def triangle(duration: int, amp: complex, period: float = None,
             phase: float = 0, name: Optional[str] = None) -> 'SamplePulse':
    """Generates triangle wave `SamplePulse`.

    Applies `midpoint` sampling strategy to generate discrete pulse from continuous function.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude. Wave range is [-amp, amp].
        period: Pulse period, units of dt. If `None` defaults to single cycle.
        phase: Pulse phase.
        name: Name of pulse.
    """
    if period is None:
        period = duration

    return _sampled_triangle_pulse(duration, amp, period, phase=phase, name=name)


_sampled_cos_pulse = samplers.midpoint(continuous.cos)


def cos(duration: int, amp: complex, freq: float = None,
        phase: float = 0, name: Optional[str] = None) -> 'SamplePulse':
    """Generates cosine wave `SamplePulse`.

    Applies `midpoint` sampling strategy to generate discrete pulse from continuous function.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude.
        freq: Pulse frequency, units of 1/dt. If `None` defaults to single cycle.
        phase: Pulse phase.
        name: Name of pulse.
    """
    if freq is None:
        freq = 1/duration

    return _sampled_cos_pulse(duration, amp, freq, phase=phase, name=name)


_sampled_sin_pulse = samplers.midpoint(continuous.sin)


def sin(duration: int, amp: complex, freq: float = None,
        phase: float = 0, name: Optional[str] = None) -> 'SamplePulse':
    """Generates sine wave `SamplePulse`.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude.
        freq: Pulse frequency, units of 1/dt. If `None` defaults to single cycle.
        phase: Pulse phase.
        name: Name of pulse.
    """
    if freq is None:
        freq = 1/duration

    return _sampled_sin_pulse(duration, amp, freq, phase=phase, name=name)


_sampled_gaussian_pulse = samplers.midpoint(continuous.gaussian)


def gaussian(duration: int, amp: complex, sigma: float, name: Optional[str] = None,
             zero_ends: bool = True) -> 'SamplePulse':
    r"""Generates unnormalized gaussian `SamplePulse`.

    Centered at `duration/2` and zeroed at `t=0` and `t=duration` to prevent large
    initial/final discontinuities.

    Applies `midpoint` sampling strategy to generate discrete pulse from continuous function.

    Integrated area under curve is $\Omega_g(amp, sigma) = amp \times np.sqrt(2\pi \sigma^2)$

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude at `duration/2`.
        sigma: Width (standard deviation) of pulse.
        name: Name of pulse.
        zero_ends: If True, make the first and last sample zero, but rescale to preserve amp.
    """
    center = duration/2
    zeroed_width = duration if zero_ends else None
    rescale_amp = bool(zero_ends)
    return _sampled_gaussian_pulse(duration, amp, center, sigma,
                                   zeroed_width=zeroed_width, rescale_amp=rescale_amp,
                                   name=name)


_sampled_gaussian_deriv_pulse = samplers.midpoint(continuous.gaussian_deriv)


def gaussian_deriv(duration: int, amp: complex, sigma: float,
                   name: Optional[str] = None) -> 'SamplePulse':
    r"""Generates unnormalized gaussian derivative `SamplePulse`.

    Applies `midpoint` sampling strategy to generate discrete pulse from continuous function.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude at `center`.
        sigma: Width (standard deviation) of pulse.
        name: Name of pulse.
    """
    center = duration/2
    return _sampled_gaussian_deriv_pulse(duration, amp, center, sigma, name=name)


_sampled_sech_pulse = samplers.midpoint(continuous.sech)


def sech(duration: int, amp: complex, sigma: float, name: str = None,
         zero_ends: bool = True) -> 'SamplePulse':
    r"""Generates unnormalized sech `SamplePulse`.

    Centered at `duration/2` and zeroed at `t=0` to prevent large initial discontinuity.

    Applies `midpoint` sampling strategy to generate discrete pulse from continuous function.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude at `duration/2`.
        sigma: Width (standard deviation) of pulse.
        name: Name of pulse.
        zero_ends: If True, make the first and last sample zero, but rescale to preserve amp.
    """
    center = duration/2
    zeroed_width = duration if zero_ends else None
    rescale_amp = bool(zero_ends)
    return _sampled_sech_pulse(duration, amp, center, sigma,
                               zeroed_width=zeroed_width, rescale_amp=rescale_amp,
                               name=name)


_sampled_sech_deriv_pulse = samplers.midpoint(continuous.sech_deriv)


def sech_deriv(duration: int, amp: complex, sigma: float, name: str = None) -> 'SamplePulse':
    r"""Generates unnormalized sech derivative `SamplePulse`.

    Applies `midpoint` sampling strategy to generate discrete pulse from continuous function.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude at `center`.
        sigma: Width (standard deviation) of pulse.
        name: Name of pulse.
    """
    center = duration/2
    return _sampled_sech_deriv_pulse(duration, amp, center, sigma, name=name)


_sampled_gaussian_square_pulse = samplers.midpoint(continuous.gaussian_square)


def gaussian_square(duration: int, amp: complex, sigma: float,
                    risefall: Optional[float] = None, width: Optional[float] = None,
                    name: Optional[str] = None, zero_ends: bool = True) -> 'SamplePulse':
    """Generates gaussian square `SamplePulse`.

    Centered at `duration/2` and zeroed at `t=0` and `t=duration` to prevent
    large initial/final discontinuities.

    Applies `midpoint` sampling strategy to generate discrete pulse from continuous function.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude.
        sigma: Width (standard deviation) of Gaussian rise/fall portion of the pulse.
        risefall: Number of samples over which pulse rise and fall happen. Width of
            square portion of pulse will be `duration-2*risefall`.
        width: The duration of the embedded square pulse. Only one of `width` or `risefall`
               should be specified since width = duration - 2 * risefall.
        name: Name of pulse.
        zero_ends: If True, make the first and last sample zero, but rescale to preserve amp.
    Raises:
        PulseError: If risefall and width arguments are inconsistent or not enough info.
    """
    if risefall is None and width is None:
        raise PulseError("gaussian_square missing required argument: 'width' or 'risefall'.")
    if risefall is not None:
        if width is None:
            width = duration - 2 * risefall
        elif 2 * risefall + width != duration:
            raise PulseError("Both width and risefall were specified, and they are "
                             "inconsistent: 2 * risefall + width == {} != "
                             "duration == {}.".format(2 * risefall + width, duration))
    center = duration / 2
    zeroed_width = duration if zero_ends else None
    return _sampled_gaussian_square_pulse(duration, amp, center, width, sigma,
                                          zeroed_width=zeroed_width, name=name)


_sampled_drag_pulse = samplers.midpoint(continuous.drag)


def drag(duration: int, amp: complex, sigma: float, beta: float,
         name: Optional[str] = None, zero_ends: bool = True) -> 'SamplePulse':
    r"""Generates Y-only correction DRAG `SamplePulse` for standard nonlinear oscillator (SNO) [1].

    Centered at `duration/2` and zeroed at `t=0` to prevent large initial discontinuity.

    Applies `midpoint` sampling strategy to generate discrete pulse from continuous function.

    [1] Gambetta, J. M., Motzoi, F., Merkel, S. T. & Wilhelm, F. K.
        Analytic control methods for high-fidelity unitary operations
        in a weakly nonlinear oscillator. Phys. Rev. A 83, 012308 (2011).

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude at `center`.
        sigma: Width (standard deviation) of pulse.
        beta: Y correction amplitude. For the SNO this is $\beta=-\frac{\lambda_1^2}{4\Delta_2}$.
            Where $\lambds_1$ is the relative coupling strength between the first excited and second
            excited states and $\Delta_2$ is the detuning between the respective excited states.
        name: Name of pulse.
        zero_ends: If True, make the first and last sample zero, but rescale to preserve amp.
    """
    center = duration/2
    zeroed_width = duration if zero_ends else None
    rescale_amp = bool(zero_ends)
    return _sampled_drag_pulse(duration, amp, center, sigma, beta,
                               zeroed_width=zeroed_width, rescale_amp=rescale_amp,
                               name=name)
