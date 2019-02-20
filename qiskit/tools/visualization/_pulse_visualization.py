# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Visualization function for pulse envelope.
"""

import numpy as np
from scipy.interpolate import CubicSpline

from qiskit.exceptions import QiskitError


def pulse_drawer(samples, duration, dt=None, interp_method='None',
                 filename=None, interactive=False,
                 dpi=150, nop=1000, size=(6, 5)):
    """Plot the interpolated envelope of pulse

    Args:
        samples (ndarray): Data points of complex pulse envelope.
        duration (int): Pulse length (number of points).
        dt (float): Time interval of samples.
        interp_method (str): Method of interpolation
            (set `None` for turn off the interpolation).
        filename (str): Name required to save pulse image.
        interactive (bool): When set true show the circuit in a new window
            (this depends on the matplotlib backend being used supporting this).
        dpi (int): Resolution of saved image.
        nop (int): Data points for interpolation.
        size (tuple): Size of figure.
    Returns:
        matplotlib.figure: A matplotlib figure object for the pulse envelope.
    Raises:
        ImportError: when the output methods requieres non-installed libraries.
        QiskitError: when invalid interpolation method is specified.
    """

    try:
        from matplotlib import pyplot as plt
    except ImportError:
        raise ImportError('pulse_drawer need matplotlib. '
                          'Run "pip install matplotlib" before.')

    if dt:
        _dt = dt
    else:
        _dt = 1

    re_y = np.real(samples)
    im_y = np.imag(samples)

    image = plt.figure(figsize=size)
    ax0 = image.add_subplot(111)

    if interp_method == 'CubicSpline':
        # spline interpolation, use mid-point of dt
        time = np.arange(0, duration + 1) * _dt + 0.5 * _dt
        cs_ry = CubicSpline(time[:-1], re_y)
        cs_iy = CubicSpline(time[:-1], im_y)

        _time = np.linspace(0, duration * _dt, nop)
        _re_y = cs_ry(_time)
        _im_y = cs_iy(_time)
    elif interp_method == 'None':
        # pseudo-DAC output
        time = np.arange(0, duration + 1) * _dt

        _time = np.r_[time[0], np.repeat(time[1:-1], 2), time[-1]]
        _re_y = np.repeat(re_y, 2)
        _im_y = np.repeat(im_y, 2)
    else:
        raise QiskitError('Invalid interpolation method "%s"' % interp_method)

    # plot
    ax0.fill_between(x=_time, y1=_re_y, y2=np.zeros_like(_time),
                     facecolor='red', alpha=0.3,
                     edgecolor='red', linewidth=1.5,
                     label='real part')
    ax0.fill_between(x=_time, y1=_im_y, y2=np.zeros_like(_time),
                     facecolor='blue', alpha=0.3,
                     edgecolor='blue', linewidth=1.5,
                     label='imaginary part')

    ax0.set_xlim(0, duration * _dt)
    ax0.grid(b=True, linestyle='-')
    ax0.legend(bbox_to_anchor=(0.5, 1.00), loc='lower center',
               ncol=2, frameon=False, fontsize=14)

    if filename:
        image.savefig(filename, dpi=dpi, bbox_inches='tight')

    plt.close(image)

    if image and interactive:
        plt.show(image)

    return image
