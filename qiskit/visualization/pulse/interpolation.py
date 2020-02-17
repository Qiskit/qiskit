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
        time (ndarray): time.
        samples (ndarray): complex pulse envelope.
        nop (int): data points for interpolation.
        kind (str): Scipy interpolation type.
            See ``scipy.interpolate.interp1d`` documentation for more information.
    Returns:
        interpolated time vector and real and imaginary part of waveform.
    """
    re_y = np.real(samples)
    im_y = np.imag(samples)

    dt = time[1] - time[0]

    time += 0.5 * dt
    cs_ry = interpolate.interp1d(time[:-1], re_y, kind=kind, bounds_error=False)
    cs_iy = interpolate.interp1d(time[:-1], im_y, kind=kind, bounds_error=False)

    time_ = np.linspace(time[0], time[-1] * dt, nop)

    return time_, cs_ry(time_), cs_iy(time_)


linear = partial(interp1d, kind='linear')
linear.__doc__ = """Apply linear interpolation between sampling points.

Args:
    time (ndarray): time.
    samples (ndarray): complex pulse envelope.
    nop (int): data points for interpolation.
Returns:
    interpolated time vector and real and imaginary part of waveform.
"""

cubic_spline = partial(interp1d, kind='cubic')
cubic_spline.__doc__ = """Apply cubic interpolation between sampling points.

Args:
    time (ndarray): time.
    samples (ndarray): complex pulse envelope.
    nop (int): data points for interpolation.
Returns:
    interpolated time vector and real and imaginary part of waveform.
"""

step_wise = partial(interp1d, kind='nearest')
step_wise.__doc__ = """No interpolation.

Args:
    time (ndarray): time.
    samples (ndarray): complex pulse envelope.
    nop (int): data points for interpolation.
Returns:
    interpolated time vector and real and imaginary part of waveform.
"""
