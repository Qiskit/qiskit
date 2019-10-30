# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-return-doc, missing-return-type-doc

"""
Pulse decorators.
"""

import functools
from typing import Callable

import numpy as np

from qiskit.pulse.exceptions import PulseError

from .sample_pulse import SamplePulse


def functional_pulse(func: Callable):
    """A decorator for generating SamplePulse from python callable.

    Args:
        func: A function describing pulse envelope.
    Raises:
        PulseError: when invalid function is specified.
    """
    @functools.wraps(func)
    def to_pulse(duration, *args, name=None, **kwargs):
        """Return SamplePulse."""
        if isinstance(duration, (int, np.integer)) and duration > 0:
            samples = func(duration, *args, **kwargs)
            samples = np.asarray(samples, dtype=np.complex128)
            return SamplePulse(samples=samples, name=name)
        raise PulseError('The first argument must be an integer value representing duration.')

    return to_pulse
