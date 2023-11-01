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

# pylint: disable=bad-docstring-quotes

"""
Deprecated.

Interpolation module for pulse visualization.
"""
from __future__ import annotations
from functools import partial

from scipy.interpolate import interp1d


linear = partial(interp1d, kind="linear")
linear.__doc__ = """Deprecated.

Apply linear interpolation between sampling points.

Args:
    time: Time vector with length of ``samples`` + 1.
    samples: Complex pulse envelope.
    nop: Number of data points for interpolation.
Returns:
    Interpolated time vector and real and imaginary part of waveform.
"""

cubic_spline = partial(interp1d, kind="cubic")
cubic_spline.__doc__ = """Deprecated.

Apply cubic interpolation between sampling points.

Args:
    time: Time vector with length of ``samples`` + 1.
    samples: Complex pulse envelope.
    nop: Number of data points for interpolation.
Returns:
    Interpolated time vector and real and imaginary part of waveform.
"""
