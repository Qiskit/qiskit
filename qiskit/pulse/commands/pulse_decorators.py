# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# TODO: pylint
# pylint: disable=redefined-builtin, missing-return-doc, missing-return-type-doc

"""
Pulse decorators.
"""

import numpy as np

from qiskit.pulse.exceptions import CommandsError
from .sample_pulse import SamplePulse


def function(func):
    """A decorator for generating SamplePulse from python callable.

    Args:
        func (callable): A function describing pulse envelope.
    Raises:
        CommandsError: when invalid function is specified.
    """

    def to_pulse(duration, *args, name=None, **kwargs):

        if isinstance(duration, int) and duration > 0:
            samples = func(duration, *args, **kwargs)
            samples = np.asarray(samples, dtype=np.complex128)
            if np.any(np.abs(samples) > 1):
                raise CommandsError('Absolute value of pulse amplitude exceeds 1.')
            return SamplePulse(samples=samples, name=name)
        raise CommandsError('The first argument must be an integer value representing duration.')

    return to_pulse
