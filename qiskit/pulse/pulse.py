# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring,missing-param-doc

"""
Pulse envelope generation.
"""

import warnings
from inspect import signature

import numpy as np

from qiskit.exceptions import QiskitError


class FunctionalPulse:
    """Pulse specification."""

    def __init__(self, pulse):
        """Register pulse envelope function.

        Args:
            pulse (callable): a function describing pulse envelope
        Raises:
            QiskitError: when incorrect envelope function is specified
        """

        if callable(pulse):
            sig = signature(pulse)
            if 'width' in sig.parameters:
                self.pulse = pulse
            else:
                raise QiskitError('Pulse function requires "width" argument.')
        else:
            raise QiskitError('Pulse function is not callable.')

    def __call__(self, width, **kwargs):
        """Return Functional Pulse with methods
        """
        return _FunctionalPulse(self.pulse, width=width, **kwargs)


class _FunctionalPulse:
    """Pulse specification with methods."""

    def __init__(self, pulse, width, **kwargs):
        """ Generate new pulse instance

        Args:
            pulse (callable): a function describing pulse envelope
            width (float): pulse width
        """
        self.pulse = pulse
        self.width = width
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

    def tolist(self):
        """Output pulse envelope as a list of complex values

        Returns:
            list: complex pulse envelope at each sampling point
        Raises:
            QiskitError: when pulse envelope is not a number
        """

        def _cmp2list(val):
            if isinstance(val, complex):
                re_v = np.real(val)
                im_v = np.imag(val)
            elif isinstance(val, float):
                re_v = val
                im_v = 0
            else:
                raise QiskitError('Pulse envelope should be numbers.')

            if np.sqrt(re_v ** 2 + im_v ** 2) > 1:
                warnings.warn('Pulse amplitude exceeds 1.')

            return [re_v, im_v]

        smp = list(map(_cmp2list, self.pulse(self.width, **self._params)))

        return smp

    def plot(self, interactive=False, **kwargs):
        """Visualize pulse envelope

        Args:
            interactive (bool): when set true show the circuit in a new window
        Returns:
            matplotlib.figure: a matplotlib figure object for the pulse envelope
        """
        from qiskit.tools.visualization._pulse_visualization import pulse_drawer

        image = pulse_drawer(np.array(self.tolist()), **kwargs)

        if image and interactive:
            image.show()

        return image
