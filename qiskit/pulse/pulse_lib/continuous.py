# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-return-doc, invalid-unary-operand-type

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
    return np.full(len(times), amp, dtype=np.complex_)


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
    return amp*(2*(2*np.floor(x) - np.floor(2*x)) + 1).astype(np.complex_)


def sawtooth(times: np.ndarray, amp: complex, period: float, phase: float = 0) -> np.ndarray:
    """Continuous sawtooth wave.

    Args:
        times: Times to output wave for.
        amp: Pulse amplitude. Wave range is [-amp, amp].
        period: Pulse period, units of dt.
        phase: Pulse phase.
    """
    x = times/period+phase/np.pi
    return amp*2*(x-np.floor(1/2+x)).astype(np.complex_)


def triangle(times: np.ndarray, amp: complex, period: float, phase: float = 0) -> np.ndarray:
    """Continuous triangle wave.

    Args:
        times: Times to output wave for.
        amp: Pulse amplitude. Wave range is [-amp, amp].
        period: Pulse period, units of dt.
        phase: Pulse phase.
    """
    return amp*(-2*np.abs(sawtooth(times, 1, period, (phase-np.pi/2)/2)) + 1).astype(np.complex_)


def cos(times: np.ndarray, amp: complex, freq: float, phase: float = 0) -> np.ndarray:
    """Continuous cosine wave.

    Args:
        times: Times to output wave for.
        amp: Pulse amplitude.
        freq: Pulse frequency, units of 1/dt.
        phase: Pulse phase.
    """
    return amp*np.cos(2*np.pi*freq*times+phase).astype(np.complex_)


def sin(times: np.ndarray, amp: complex, freq: float, phase: float = 0) -> np.ndarray:
    """Continuous cosine wave.

    Args:
        times: Times to output wave for.
        amp: Pulse amplitude.
        freq: Pulse frequency, units of 1/dt.
        phase: Pulse phase.
    """
    return amp*np.sin(2*np.pi*freq*times+phase).astype(np.complex_)


def _fix_gaussian_width(gaussian_samples, amp: float, center: float, sigma: float,
                        zeroed_width: Union[None, float] = None, rescale_amp: bool = False,
                        ret_scale_factor: bool = False) -> np.ndarray:
    r"""Enforce that the supplied gaussian pulse is zeroed at a specific width.

    This is acheived by subtracting $\Omega_g(center \pm zeroed_width/2)$ from all samples.

    amp: Pulse amplitude at `2\times center+1`.
    center: Center (mean) of pulse.
    sigma: Width (standard deviation) of pulse.
    zeroed_width: Subtract baseline to gaussian pulses to make sure
             $\Omega_g(center \pm zeroed_width/2)=0$ is satisfied. This is used to avoid
             large discontinuities at the start of a gaussian pulse. If unsupplied,
             defaults to $2*(center+1)$ such that the samples are zero at $\Omega_g(-1)$.
    rescale_amp: If `zeroed_width` is not `None` and `rescale_amp=True` the pulse will
                 be rescaled so that $\Omega_g(center)-\Omega_g(center\pm zeroed_width/2)=amp$.
    ret_scale_factor: Return amplitude scale factor.
    """
    if zeroed_width is None:
        zeroed_width = 2*(center+1)

    zero_offset = gaussian(np.array([-zeroed_width/2]), amp, center, sigma)
    gaussian_samples -= zero_offset
    amp_scale_factor = 1.
    if rescale_amp:
        amp_scale_factor = amp/(amp-zero_offset) if amp-zero_offset != 0 else 1.
        gaussian_samples *= amp_scale_factor

    if ret_scale_factor:
        return gaussian_samples, amp_scale_factor
    return gaussian_samples


def gaussian(times: np.ndarray, amp: complex, center: float, sigma: float,
             zeroed_width: Union[None, float] = None, rescale_amp: bool = False,
             ret_x: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    r"""Continuous unnormalized gaussian pulse.

    Integrated area under curve is $\Omega_g(amp, sigma) = amp \times np.sqrt(2\pi \sigma^2)$

    Args:
        times: Times to output pulse for.
        amp: Pulse amplitude at `center`. If `zeroed_width` is set pulse amplitude at center
            will be $amp-\Omega_g(center\pm zeroed_width/2)$ unless `rescale_amp` is set,
            in which case all samples will be rescaled such that the center
            amplitude will be `amp`.
        center: Center (mean) of pulse.
        sigma: Width (standard deviation) of pulse.
        zeroed_width: Subtract baseline to gaussian pulses to make sure
                 $\Omega_g(center \pm zeroed_width/2)=0$ is satisfied. This is used to avoid
                 large discontinuities at the start of a gaussian pulse.
        rescale_amp: If `zeroed_width` is not `None` and `rescale_amp=True` the pulse will
                     be rescaled so that $\Omega_g(center)-\Omega_g(center\pm zeroed_width/2)=amp$.
        ret_x: Return centered and standard deviation normalized pulse location.
               $x=(times-center)/sigma.
    """
    times = np.asarray(times, dtype=np.complex_)
    x = (times-center)/sigma
    gauss = amp*np.exp(-x**2/2).astype(np.complex_)

    if zeroed_width is not None:
        gauss = _fix_gaussian_width(gauss, amp=amp, center=center, sigma=sigma,
                                    zeroed_width=zeroed_width, rescale_amp=rescale_amp)

    if ret_x:
        return gauss, x
    return gauss


def gaussian_deriv(times: np.ndarray, amp: complex, center: float, sigma: float,
                   ret_gaussian: bool = False) -> np.ndarray:
    """Continuous unnormalized gaussian derivative pulse.

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
                    sigma: float, zeroed_width: Union[None, float] = None) -> np.ndarray:
    r"""Continuous gaussian square pulse.

    Args:
        times: Times to output pulse for.
        amp: Pulse amplitude.
        center: Center of the square pulse component.
        width: Width of the square pulse component.
        sigma: Width (standard deviation) of gaussian rise/fall portion of the pulse.
        zeroed_width: Subtract baseline of gaussian square pulse
                      to enforce $\OmegaSquare(center \pm zeroed_width/2)=0$.
    """
    square_start = center-width/2
    square_stop = center+width/2
    if zeroed_width:
        zeroed_width = min(width, zeroed_width)
        gauss_zeroed_width = zeroed_width-width
    else:
        gauss_zeroed_width = None

    funclist = [functools.partial(gaussian, amp=amp, center=square_start, sigma=sigma,
                                  zeroed_width=gauss_zeroed_width, rescale_amp=True),
                functools.partial(gaussian, amp=amp, center=square_stop, sigma=sigma,
                                  zeroed_width=gauss_zeroed_width, rescale_amp=True),
                functools.partial(constant, amp=amp)]
    condlist = [times <= square_start, times >= square_stop]
    return np.piecewise(times.astype(np.complex_), condlist, funclist)


def drag(times: np.ndarray, amp: complex, center: float, sigma: float, beta: float,
         zeroed_width: Union[None, float] = None, rescale_amp: bool = False) -> np.ndarray:
    r"""Continuous Y-only correction DRAG pulse for standard nonlinear oscillator (SNO) [1].

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
        zeroed_width: Subtract baseline to gaussian pulses to make sure
                 $\Omega_g(center \pm zeroed_width/2)=0$ is satisfied. This is used to avoid
                 large discontinuities at the start of a gaussian pulse.
        rescale_amp: If `zeroed_width` is not `None` and `rescale_amp=True` the pulse will
                     be rescaled so that $\Omega_g(center)-\Omega_g(center\pm zeroed_width/2)=amp$.

    """
    gauss_deriv, gauss = gaussian_deriv(times, amp=amp, center=center, sigma=sigma,
                                        ret_gaussian=True)
    if zeroed_width is not None:
        gauss, scale_factor = _fix_gaussian_width(gauss, amp=amp, center=center, sigma=sigma,
                                                  zeroed_width=zeroed_width,
                                                  rescale_amp=rescale_amp,
                                                  ret_scale_factor=True)
        gauss_deriv *= scale_factor

    return gauss + 1j*beta*gauss_deriv
