# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
interpolation module for pulse visualization.
"""

from scipy.interpolate import CubicSpline
import numpy as np


def cubic_spline(time, samples, nop):
    """Cubic spline interpolation.

    Args:
        time (ndarray): time.
        samples (ndarray): complex pulse envelope.
        nop (int): data points for interpolation.

    Returns:
        ndarray: interpolated waveform.
    """
    re_y = np.real(samples)
    im_y = np.imag(samples)

    dt = time[1] - time[0]

    time += 0.5 * dt
    cs_ry = CubicSpline(time[:-1], re_y)
    cs_iy = CubicSpline(time[:-1], im_y)

    time_ = np.linspace(time[0], time[-1] * dt, nop)

    return time_, cs_ry(time_), cs_iy(time_)


def step_wise(time, samples, nop):
    """Step-wise interpolation.

    Args:
        time (ndarray): time.
        samples (ndarray): complex pulse envelope.
        nop (int): data points for interpolation.

    Returns:
        ndarray: interpolated waveform.
    """
    # pylint: disable=unused-argument

    re_y = np.real(samples)
    im_y = np.imag(samples)

    re_y = np.repeat(re_y, 2)
    im_y = np.repeat(im_y, 2)

    time_ = np.r_[time[0], np.repeat(time[1:-1], 2), time[-1]]

    return time_, re_y, im_y
