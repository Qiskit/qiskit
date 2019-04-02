# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-return-doc

"""Module for builtin continuous pulse functions."""

import functools
from typing import Union, Tuple

import numpy as np


def constant(times: np.ndarray, amp: complex) -> np.ndarray:
    """Continuous constant pulse.

    Args:
        times: Times to output pulse for.
        amp: Complex pulse amplitude.
    """
    return np.full(len(times), amp, dtype=np.complex)


def zero(times: np.ndarray) -> np.ndarray:
    """Continuous zero pulse.

    Args:
        times: Times to output pulse for.
    """
    return constant(times, 0)


def square(times: np.ndarray, amp: complex, period: float, phase: float = 0) -> np.ndarray:
    """Continuous square wave.

    Args:
        times: Times to output wave for.
        amp: Pulse amplitude. Wave range is [-amp, amp].
        period: Pulse period, units of dt.
        phase: Pulse phase.
    """
    x = times/period+phase/np.pi
    return amp*(2*(2*np.floor(x) - np.floor(2*x)) + 1)


def sawtooth(times: np.ndarray, amp: complex, period: float, phase: float = 0) -> np.ndarray:
    """Continuous sawtooth wave.

    Args:
        times: Times to output wave for.
        amp: Pulse amplitude. Wave range is [-amp, amp].
        period: Pulse period, units of dt.
        phase: Pulse phase.
    """
    x = times/period+phase/np.pi
    return amp*2*(x-np.floor(1/2+x))


def triangle(times: np.ndarray, amp: complex, period: float, phase: float = 0) -> np.ndarray:
    """Continuous triangle wave.

    Args:
        times: Times to output wave for.
        amp: Pulse amplitude. Wave range is [-amp, amp].
        period: Pulse period, units of dt.
        phase: Pulse phase.
    """
    return amp*(-2*np.abs(sawtooth(times, 1, period, (phase-np.pi/2)/2)) + 1)


def cos(times: np.ndarray, amp: complex, freq: float, phase: float = 0) -> np.ndarray:
    """Continuous cosine wave.

    Args:
        times: Times to output wave for.
        amp: Pulse amplitude.
        freq: Pulse frequency, units of 1/dt.
        phase: Pulse phase.
    """
    return amp*np.cos(2*np.pi*freq*times+phase)


def sin(times: np.ndarray, amp: complex, freq: float, phase: float = 0) -> np.ndarray:
    """Continuous cosine wave.

    Args:
        times: Times to output wave for.
        amp: Pulse amplitude.
        freq: Pulse frequency, units of 1/dt.
        phase: Pulse phase.
    """
    return amp*np.sin(2*np.pi*freq*times+phase)


def _zero_gaussian_at(gaussian_samples, amp: float, center: float, sigma: float,
                      zero_at: float = -1, rescale_amp: bool = False,
                      ret_scale_factor: bool = False) -> np.ndarray:
    """Zero and optionally rescale a gaussian pulse.
    amp: Pulse amplitude at `center`.
    center: Center (mean) of pulse.
    sigma: Width (standard deviation) of pulse.
    zero_at: Subtract baseline to gaussian pulses to make sure $\Omega_g(zero_at)=0$ is
             satisfied. Note this will also cause $\Omega_g(2*center-zero_at)=0$.
             This is used to avoid large discontinuities at the start of the gaussian pulse.
    rescale_amp: If `zero_at=True` and `rescale_amp=True` the pulse will be rescaled so that
                 $\Omega_g(center)-\Omega_g(zero_at)=amp$.
    ret_scale_factor: Return amplitude scale factor.
    """
    zero_offset = gaussian(np.array([zero_at]), amp, center, sigma)
    gaussian_samples -= zero_offset
    amp_scale_factor = 1.
    if rescale_amp:
        amp_scale_factor = amp/(amp-zero_offset)
        gaussian_samples *= amp_scale_factor

    if ret_scale_factor:
        return gaussian_samples, amp_scale_factor
    return gaussian_samples


def gaussian(times: np.ndarray, amp: complex, center: float, sigma: float,
             zero_at: Union[None, int] = None, rescale_amp: bool = False,
             ret_x: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    r"""Continuous unnormalized gaussian pulse.

    Integrated area under curve is $\Omega_g(amp, sigma) = amp \times np.sqrt(2\pi \sigma^2)$

    Args:
        times: Times to output pulse for.
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
    x = (times-center)/sigma
    gauss = amp*np.exp(-x**2/2)

    if zero_at is not None:
        gauss = _zero_gaussian_at(gauss, amp=amp, center=center, sigma=sigma,
                                  zero_at=zero_at, rescale_amp=rescale_amp)

    if ret_x:
        return gauss, x
    return gauss


def gaussian_deriv(times: np.ndarray, amp: complex, center: float, sigma: float,
                   ret_gaussian: bool = False) -> np.ndarray:
    r"""Continuous unnormalized gaussian derivative pulse.

    Args:
        times: Times to output pulse for.
        amp: Pulse amplitude at `center`.
        center: Center (mean) of pulse.
        sigma: Width (standard deviation) of pulse.
        ret_gaussian: Return gaussian with which derivative was taken with.
    """
    gauss, x = gaussian(times, amp=amp, center=center, sigma=sigma, ret_x=True)
    gauss_deriv = -x/sigma*gauss
    if ret_gaussian:
        return gauss_deriv, gauss
    return gauss_deriv


def gaussian_square(times: np.ndarray, amp: complex, center: float, width: float,
                    sigma: float, rise_zero_at: Union[None, int] = None,
                    fall_zero_at: Union[None, int] = None) -> np.ndarray:
    """Continuous gaussian square pulse.

    Rise of pulse is gaussian followed by square pulse and finall guassian fall.

    Args:
        times: Times to output pulse for.
        amp: Pulse amplitude.
        center: Center of the square pulse component.
        width: Width of the square pulse component.
        sigma: Width (standard deviation) of gaussian rise/fall portion of the pulse.
        rise_zero_at: Subtract baseline to gaussian rise pulse
                      to enforce $\Omega_rise(rise_zero_at)=0$.
        fall_zero_at: Subtract baseline to gaussian fall pulse
                      to enforce $\Omega_fall(fall_zero_at)=0$.
    """
    square_start = center-width/2
    square_stop = center+width/2
    funclist = [functools.partial(gaussian, amp=amp, center=square_start, sigma=sigma,
                                  zero_at=rise_zero_at, rescale_amp=True),
                functools.partial(gaussian, amp=amp, center=square_stop, sigma=sigma,
                                  zero_at=fall_zero_at, rescale_amp=True),
                functools.partial(constant, amp=amp)]
    condlist = [times <= square_start, times >= square_stop]
    return np.piecewise(times, condlist, funclist)


def drag(times: np.ndarray, amp: complex, center: float, sigma: float, beta: float,
         zero_at: Union[None, int] = None, rescale_amp: bool = False) -> np.ndarray:
    r"""Continuous Y-only correction DRAG pulse for standard nonlinear oscilattor (SNO) [1].

    [1] Gambetta, J. M., Motzoi, F., Merkel, S. T. & Wilhelm, F. K.
        Analytic control methods for high-fidelity unitary operations
        in a weakly nonlinear oscillator. Phys. Rev. A 83, 012308 (2011).


    Args:
        times: Times to output pulse for.
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
    gauss_deriv, gauss = gaussian_deriv(times, amp=amp, center=center, sigma=sigma,
                                        ret_gaussian=True)
    if zero_at is not None:
        gauss, scale_factor = _zero_gaussian_at(gauss, amp=amp, center=center, sigma=sigma,
                                                zero_at=zero_at, rescale_amp=rescale_amp,
                                                ret_scale_factor=True)
        gauss_deriv *= scale_factor

    return gauss + 1j*beta*gauss_deriv
