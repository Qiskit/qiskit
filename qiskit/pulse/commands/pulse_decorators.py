# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-return-doc, missing-return-type-doc

"""
Pulse decorators.
"""

import functools
import numpy as np

from qiskit.pulse.exceptions import PulseError
from .sample_pulse import SamplePulse


def functional_pulse(func):
    """A decorator for generating SamplePulse from python callable.
    Args:
        func (callable): A function describing pulse envelope.
    Raises:
        PulseError: when invalid function is specified.
    """
    @functools.wraps(func)
    def to_pulse(duration, *args, name=None, **kwargs):
        """Return SamplePulse."""
        if isinstance(duration, int) and duration > 0:
            samples = func(duration, *args, **kwargs)
            samples = np.asarray(samples, dtype=np.complex128)
            return SamplePulse(samples=samples, name=name)
        raise PulseError('The first argument must be an integer value representing duration.')

    return to_pulse
