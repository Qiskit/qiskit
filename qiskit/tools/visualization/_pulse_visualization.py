# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring,missing-param-doc

"""
Visualization function for pulse envelope.
"""

import numpy as np
from scipy.interpolate import CubicSpline


def pulse_drawer(samples, duration, dt=None, interp_method='CubicSpline',
                 filename=None, interactive=False,
                 dpi=150, nop=1000, size=(6, 5)):
    """Plot the interpolated envelope of pulse

    Args:
        samples (ndarray): data points of complex pulse envelope
        duration (int): pulse length (number of points)
        dt (float): time interval of samples
        interp_method (str): method of interpolation
            (set `None` for turn off the interpolation)
        filename (str): name required to save pulse image
        interactive (bool): when set true show the circuit in a new window
            (for `mpl` this depends on the matplotlib backend being used
            supporting this).
        dpi (int): resolution of saved image
        nop (int): data points for interpolation
        size (tuple): size of figure
    Returns:
        matplotlib.figure: a matplotlib figure object for the pulse envelope
    Raises:
        ImportError: when the output methods requieres non-installed libraries.
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
    time = np.arange(0, duration + 1) * _dt

    # pseudo-AWG output
    _re_y = np.repeat(re_y, 2)
    _im_y = np.repeat(im_y, 2)
    _time = np.r_[time[0], np.repeat(time[1:-1], 2), time[-1]]

    image = plt.figure(figsize=size)
    ax0 = image.add_subplot(111)

    ax0.plot(_time, _re_y, color='red',
             linewidth=1.5, label='real part')
    ax0.plot(_time, _im_y, color='blue',
             linewidth=1.5, label='imaginary part')
    ax0.scatter(x=time[:-1], y=re_y, color='red', marker='o')
    ax0.scatter(x=time[:-1], y=im_y, color='blue', marker='o')

    if interp_method == 'CubicSpline':
        # spline interpolation
        cs_ry = CubicSpline(time[:-1], re_y)
        cs_iy = CubicSpline(time[:-1], im_y)
        time_interp = np.linspace(0, duration * _dt, nop)
        # plot
        ax0.fill_between(x=time_interp, y1=cs_ry(time_interp),
                         y2=np.zeros_like(time_interp),
                         facecolors='red', alpha=0.1)
        ax0.fill_between(x=time_interp, y1=cs_iy(time_interp),
                         y2=np.zeros_like(time_interp),
                         facecolors='blue', alpha=0.1)

    ax0.set_xlim(0, max(time))
    ax0.grid(b=True, linestyle='-')
    ax0.legend(bbox_to_anchor=(0.5, 1.00), loc='lower center',
               ncol=2, frameon=False, fontsize=14)

    if filename:
        image.savefig(filename, dpi=dpi, bbox_inches='tight')

    plt.close(image)

    if image and interactive:
        plt.show(image)

    return image
