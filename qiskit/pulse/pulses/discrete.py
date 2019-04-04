# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-return-doc

"""Module for builtin discrete pulses."""

from typing import Union, Tuple

import numpy as np

from qiskit.pulse.commands import SamplePulse
from qiskit.pulse.pulses import continuous


def constant(duration: int, amp: complex) -> SamplePulse:
    """Generates constant-sampled `SamplePulse`.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Complex pulse amplitude.
    """


def zero(duration: int) -> SamplePulse:
    """Generates zero-sampled `SamplePulse`.

    Args:
        duration: Duration of pulse. Must be greater than zero.
    """


def square(duration: int, amp: complex, period: float, phase: float = 0) -> SamplePulse:
    """Generates squaure wave `SamplePulse`.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude. Wave range is [-amp, amp].
        period: Pulse period, units of dt.
        phase: Pulse phase.
    """


def sawtooth(duration: int, amp: complex, period: float, phase: float = 0) -> SamplePulse:
    """Generates sawtooth wave `SamplePulse`.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude. Wave range is [-amp, amp].
        period: Pulse period, units of dt.
        phase: Pulse phase.
    """


def triangle(duration: int, amp: complex, period: float, phase: float = 0) -> SamplePulse:
    """Generates triangle wave `SamplePulse`.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude. Wave range is [-amp, amp].
        period: Pulse period, units of dt.
        phase: Pulse phase.
    """


def cos(duration: int, amp: complex, freq: float, phase: float = 0) -> SamplePulse:
    """Generates cosine wave `SamplePulse`.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude.
        freq: Pulse frequency, units of 1/dt.
        phase: Pulse phase.
    """


def sin(duration: int, amp: complex, freq: float, phase: float = 0) -> SamplePulse:
    """Generates sine wave `SamplePulse`.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude.
        freq: Pulse frequency, units of 1/dt.
        phase: Pulse phase.
    """


def gaussian(duration: int, amp: complex, center: float, sigma: float,
             zero_at: Union[None, int] = None, rescale_amp: bool = False,
             ret_x: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    r"""Generates unnormalized gaussian `SamplePulse`.

    Integrated area under curve is $\Omega_g(amp, sigma) = amp \times np.sqrt(2\pi \sigma^2)$

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude at `center`.
        center: Center (mean) of pulse.
        sigma: Width (standard deviation) of pulse.
        zero_at: Subtract baseline to gaussian pulses to make sure $\Omega_g(zero_at)=0$ is
                 satisfied. Note this will also cause $\Omega_g(2*center-zero_at)=0$.
                 This is used to avoid large discontinuities at the start of the gaussian pulse.
        rescale_amp: If `zero_at=True` and `rescale_amp=True` the pulse will be rescaled so that
                     $\Omega_g(center)-\Omega_g(zero_at)=amp$.
        ret_x: Return centered and standard deviation normalized pulse location.
               $x=(times-center)/sigma.
    """


def gaussian_deriv(duration: int, amp: complex, center: float, sigma: float,
                   ret_gaussian: bool = False) -> SamplePulse:
    r"""Generates unnormalized gaussian derivative `SamplePulse`.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude at `center`.
        center: Center (mean) of pulse.
        sigma: Width (standard deviation) of pulse.
        ret_gaussian: Return gaussian with which derivative was taken with.
    """


def gaussian_square(duration: int, amp: complex, center: float, width: float,
                    sigma: float, rise_zero_at: Union[None, int] = None,
                    fall_zero_at: Union[None, int] = None) -> SamplePulse:
    """Generates gaussian square `SamplePulse`.

    Rise of pulse is gaussian followed by square pulse and finall guassian fall.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude.
        center: Center of the square pulse component.
        width: Width of the square pulse component.
        sigma: Width (standard deviation) of gaussian rise/fall portion of the pulse.
        rise_zero_at: Subtract baseline to gaussian rise pulse
                      to enforce $\Omega_rise(rise_zero_at)=0$.
        fall_zero_at: Subtract baseline to gaussian fall pulse
                      to enforce $\Omega_fall(fall_zero_at)=0$.
    """


def drag(duration: int, amp: complex, center: float, sigma: float, beta: float,
         zero_at: Union[None, int] = None, rescale_amp: bool = False) -> SamplePulse:
    r"""Generates Y-only correction DRAG `SamplePulse` for standard nonlinear oscilattor (SNO) [1].

    [1] Gambetta, J. M., Motzoi, F., Merkel, S. T. & Wilhelm, F. K.
        Analytic control methods for high-fidelity unitary operations
        in a weakly nonlinear oscillator. Phys. Rev. A 83, 012308 (2011).


    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude at `center`.
        center: Center (mean) of pulse.
        sigma: Width (standard deviation) of pulse.
        beta: Y correction amplitude. For the SNO this is $\beta=-\frac{\lambda_1^2}{4\Delta_2}$.
            Where $\lambds_1$ is the relative coupling strength between the first excited and second
            excited states and $\Delta_2$ is the detuning between the resepective excited states.
        zero_at: Subtract baseline to gaussian pulses to make sure $\Omega_g(zero_at)=0$ is
                 satisfied. Note this will also cause $\Omega_g(2*center-zero_at)=0$.
                 This is used to avoid large discontinuities at the start of the gaussian pulse.
        rescale_amp: If `zero_at=True` and `rescale_amp=True` the pulse will be rescaled so that
                     $\Omega_g(center)-\Omega_g(zero_at)=amp$.

    """
