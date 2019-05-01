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

from scipy import interpolate
import numpy as np


def interp1d(time, samples, nop, kind='linear'):
    """Scipy interpolation wrapper.

    Args:
        time (ndarray): time.
        samples (ndarray): complex pulse envelope.
        nop (int): data points for interpolation.
        kind (str): Scipy interpolation type. See `scipy.interpolate.interp1d` documentation
            for more information.
    Returns:
        ndarray: interpolated waveform.
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

cubic_spline = partial(interp1d, kind='cubic')

step_wise = partial(interp1d, kind='nearest')
