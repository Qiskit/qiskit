# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-return-doc

"""Module for builtin continuous pulse functions."""

import functools

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


def gaussian(times: np.ndarray, amp: complex, center: float, sigma: float) -> np.ndarray:
    """Continuous gaussian wave.

    Args:
        times: Times to output pulse for.
        amp: Pulse amplitude.
        center: Center (mean) of pulse.
        sigma: Width (standard deviation) of pulse.
    """
    x = (times-center)/sigma
    return amp*np.exp(-x**2/2)/np.sqrt(2*np.pi)


def gaussian_square(times: np.ndarray, amp: complex, center: float, width: float,
                    sigma: float) -> np.ndarray:
    """Continuous gaussian square pulse.

    Rise of pulse is gaussian followed by square pulse and finall guassian fall.

    Args:
        times: Times to output pulse for.
        amp: Pulse amplitude.
        center: Center of the square pulse component.
        width: Width of the square pulse component.
        sigma: Width (standard deviation) of gaussian rise/fall portion of the pulse.
    """
    square_start = center-width/2
    square_stop = center+width/2
    funclist = [functools.partial(gaussian, amp=gaussian, center=square_start, sigma=sigma),
                functools.partial(gaussian, amp=gaussian, center=square_stop, sigma=sigma),
                functools.partial(constant, amp=amp)]
    condlist = [times <= square_start, times >= square_stop]
    return np.piecewise(times, condlist, funclist)
