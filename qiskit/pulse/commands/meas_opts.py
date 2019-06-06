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

# pylint: disable=missing-param-doc,useless-super-delegation

"""
Measurement options.
"""


class MeasOpts:
    """Measurement options."""

    def __init__(self, name=None, **params):
        """Create new measurement options.

        Parameters:
            name (str): Name of measurement option to be used.
        """
        self._name = name
        self._params = params

    @property
    def name(self):
        """Return parameter name."""
        return self._name

    @property
    def params(self):
        """Return parameter dict."""
        return self._params

    def __eq__(self, other):
        """Two measurement options are the same if they are of the same type
        and have the same name and params.

        Args:
            other (MeasOpts): Other Discriminator/Kernel.

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self._name == other._name and \
                self._params == other._params:
            return True
        return False

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self._name)


class Discriminator(MeasOpts):
    """Discriminator."""

    def __init__(self, name=None, **params):
        """Create new discriminator.

        Parameters:
            name (str): Name of discriminator to be used.
        """
        super().__init__(name, **params)


class Kernel(MeasOpts):
    """Kernel."""

    def __init__(self, name=None, **params):
        """Create new kernel.

        Parameters:
            name (str): Name of kernel to be used.
        """
        super().__init__(name, **params)
