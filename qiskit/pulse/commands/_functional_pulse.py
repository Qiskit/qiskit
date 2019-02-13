# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-param-doc

"""
Functional pulse.
"""

import warnings
from inspect import signature

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.pulse.commands._sample_pulse import SamplePulse


class FunctionalPulse:
    """Decorator of pulse envelope function."""

    def __init__(self, pulse):
        """Create new pulse envelope function.

        Args:
            pulse (callable): a function describing pulse envelope.
            pulse function must contain argument "duration".

        Raises:
            QiskitError: when incorrect envelope function is specified
        """

        if callable(pulse):
            sig = signature(pulse)
            if 'duration' in sig.parameters:
                self.pulse = pulse
            else:
                raise QiskitError('Pulse function requires "duration" argument.')
        else:
            raise QiskitError('Pulse function is not callable.')

    def __call__(self, duration, **kwargs):
        """Create new functional pulse.
        """
        return FunctionalPulseComm(self.pulse, duration=duration, **kwargs)


class FunctionalPulseComm(SamplePulse):
    """Functional pulse."""

    def __init__(self, pulse, duration, **kwargs):
        """ Generate new pulse instance.

        Args:
            pulse (callable): a function describing pulse envelope
            duration (int): duration of pulse
        """

        super(FunctionalPulseComm, self).__init__(duration)

        self.pulse = pulse
        self._params = kwargs

    @property
    def params(self):
        """Get parameters for describing pulse envelope

        Returns:
            dict: pulse parameters
        """
        return self._params

    @params.setter
    def params(self, params_new):
        """Set parameters for describing pulse envelope

        Args:
            params_new (dict): dictionary of parameters
        Raises:
            QiskitError: when pulse parameter is not in the correct format.
        """
        if isinstance(params_new, dict):
            for key, val in self._params.items():
                self._params[key] = params_new.get(key, val)
        else:
            raise QiskitError('Pulse parameter should be dictionary.')

    @property
    def sample(self):
        """Output pulse envelope as a list of complex values

        Returns:
            list: complex pulse envelope at each sampling point
        Raises:
            QiskitError: when invalid pulse data is generated
        """
        samples = self.pulse(self.duration, **self._params)

        if not isinstance(samples, (list, np.ndarray)):
            raise QiskitError('Output from pulse function is not array.')

        if len(samples) != self.duration:
            raise QiskitError('Number of Data point is not consistent with duration.')

        if any(abs(samples) > 1):
            warnings.warn("Absolute value of pulse amplitude exceeds 1.")
            _samples = np.where(abs(samples) > 1, samples/abs(samples), samples)
        else:
            _samples = samples

        return _samples
