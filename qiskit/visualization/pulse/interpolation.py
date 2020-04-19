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

# pylint: disable=invalid-name

"""
interpolation module for pulse visualization.
"""
from functools import partial
from typing import Tuple

import numpy as np
from scipy import interpolate


def interp1d(time: np.ndarray,
             samples: np.ndarray,
             nop: int, kind: str = 'linear'
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scipy interpolation wrapper.

    Args:
        time: Time vector with length of ``samples`` + 1.
        samples: Complex pulse envelope.
        nop: Number of data points for interpolation.
        kind: Scipy interpolation type.
            See ``scipy.interpolate.interp1d`` documentation for more information.
    Returns:
        Interpolated time vector and real and imaginary part of waveform.
    """
    re_y = np.real(samples)
    im_y = np.imag(samples)

    dt = time[1] - time[0]

    time += 0.5 * dt
    cs_ry = interpolate.interp1d(
        time[:-1], re_y, kind=kind, bounds_error=False)
    cs_iy = interpolate.interp1d(
        time[:-1], im_y, kind=kind, bounds_error=False)

    time_ = np.linspace(time[0], time[-1] * dt, nop)

    return time_, cs_ry(time_), cs_iy(time_)


def step_wise(time: np.ndarray,
              samples: np.ndarray,
              nop: int
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # pylint: disable=unused-argument
    """Keep uniform variation between sample values. No interpolation is applied.
    Args:
        time: Time vector with length of ``samples`` + 1.
        samples: Complex pulse envelope.
        nop: This argument is not used.
    Returns:
        Time vector and real and imaginary part of waveform.
    """
    samples_ = np.repeat(samples, 2)
    re_y_ = np.real(samples_)
    im_y_ = np.imag(samples_)
    time__ = np.concatenate(([time[0]], np.repeat(time[1:-1], 2), [time[-1]]))
    return time__, re_y_, im_y_


linear = partial(interp1d, kind='linear')
linear.__doc__ = """Apply linear interpolation between sampling points.

Args:
    time: Time vector with length of ``samples`` + 1.
    samples: Complex pulse envelope.
    nop: Number of data points for interpolation.
Returns:
    Interpolated time vector and real and imaginary part of waveform.
"""

cubic_spline = partial(interp1d, kind='cubic')
cubic_spline.__doc__ = """Apply cubic interpolation between sampling points.

Args:
    time: Time vector with length of ``samples`` + 1.
    samples: Complex pulse envelope.
    nop: Number of data points for interpolation.
Returns:
    Interpolated time vector and real and imaginary part of waveform.
"""
