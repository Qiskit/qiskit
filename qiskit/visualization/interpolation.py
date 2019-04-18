# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

from scipy.interpolate import CubicSpline
import numpy as np


def cubic_spline(samples, dt, nop):
    """Cubic spline interpolation.

    Args:
        samples (ndarray): complex pulse envelope.
        dt (float): time interval of samples.
        nop (int): data points for interpolation.

    Returns:
        ndarray: interpolated waveform.
    """
    re_y = np.real(samples)
    im_y = np.imag(samples)

    time = (np.arange(0, len(samples) + 1) + 0.5) * dt
    cs_ry = CubicSpline(time[:-1], re_y)
    cs_iy = CubicSpline(time[:-1], im_y)

    time_ = np.linspace(0, len(samples) * dt, nop)

    return time_, cs_ry(time_), cs_iy(time_)


def step_wise(samples, dt, nop):
    """Step-wise interpolation.

    Args:
        samples (ndarray): complex pulse envelope.
        dt (float): time interval of samples.
        nop (int): data points for interpolation.

    Returns:
        ndarray: interpolated waveform.
    """
    re_y = np.real(samples)
    im_y = np.imag(samples)

    time = np.arange(0, len(samples) + 1) * dt
    re_y = np.repeat(re_y, 2)
    im_y = np.repeat(im_y, 2)

    time_ = np.r_[time[0], np.repeat(time[1:-1], 2), time[-1]]

    return time_, re_y, im_y
