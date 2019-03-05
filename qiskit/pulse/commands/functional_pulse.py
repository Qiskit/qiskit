# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-param-doc

"""
Functional pulse.
"""

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.pulse.commands.sample_pulse import SamplePulse


class FunctionalPulse:
    """Decorator of pulse envelope function."""

    def __init__(self, pulse_fun):
        """Create new pulse envelope function.

        Args:
            pulse_fun (callable): A function describing pulse envelope.
            Pulse function must contain argument "duration".

        Raises:
            QiskitError: when incorrect envelope function is specified.
        """

        if callable(pulse_fun):
            self.pulse_fun = pulse_fun
        else:
            raise QiskitError('Pulse function is not callable.')

    def __call__(self, duration, name=None, **params):
        """Create new functional pulse.
        """
        return FunctionalPulseCommand(self.pulse_fun, duration=duration, name=name, **params)


class FunctionalPulseCommand(SamplePulse):
    """Functional pulse."""

    def __init__(self, pulse_fun, duration, name, **params):
        """Generate new pulse instance.

        Args:
            pulse_fun (callable): A function describing pulse envelope.
            duration (int): Duration of pulse.
            name (str): Unique name to identify the pulse.
        Raises:
            QiskitError: when first argument of pulse function is not duration.
        """

        if isinstance(duration, int) and duration > 0:
            super(FunctionalPulseCommand, self).__init__(duration=duration,
                                                         samples=None,
                                                         name=name)

            self.pulse_fun = pulse_fun
            self._params = params
        else:
            raise QiskitError('The first argument of pulse function must be duration.')

    @property
    def params(self):
        """Get parameters for describing pulse envelope.

        Returns:
            dict: Pulse parameters.
        """
        return self._params

    def update_params(self, **params):
        """Set parameters for describing pulse envelope.
        """
        self._params.update(params)

    @property
    def samples(self):
        """Output pulse envelope as a list of complex values.

        Returns:
            ndarray: Complex array of pulse envelope.
        Raises:
            QiskitError: when invalid pulse data is generated.
        """
        samples = self.pulse_fun(self.duration, **self.params)

        _samples = np.asarray(samples, dtype=np.complex128)

        if len(_samples) != self.duration:
            raise QiskitError('Number of Data point is not consistent with duration.')

        if np.any(np.abs(_samples) > 1):
            raise QiskitError('Absolute value of pulse amplitude exceeds 1.')

        return _samples
