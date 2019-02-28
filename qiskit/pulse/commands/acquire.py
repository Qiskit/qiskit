# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-param-doc,useless-super-delegation

"""
Acquire.
"""

from qiskit.exceptions import QiskitError
from qiskit.pulse.commands.meas_opts import MeasOpts
from qiskit.pulse.commands.pulse_command import PulseCommand


class Acquire(PulseCommand):
    """Acquire."""

    def __init__(self, duration, discriminator=None, kernel=None):
        """Create new acquire command.

        Args:
            duration (int): Duration of acquisition.
            discriminator (Discriminator): Discriminators to be used
                (from the list of available discriminator) if the measurement level is 2.
            kernel (Kernel): The data structures defining the measurement kernels
                to be used (from the list of available kernels) and set of parameters
                (if applicable) if the measurement level is 1 or 2.

        Raises:
            QiskitError: when invalid discriminator or kernel object is input.
        """

        super(Acquire, self).__init__(duration=duration, name='acquire')

        if discriminator:
            if isinstance(discriminator, Discriminator):
                self.discriminator = discriminator
            else:
                raise QiskitError('Invalid discriminator object is specified.')
        else:
            self.discriminator = Discriminator()

        if kernel:
            if isinstance(kernel, Kernel):
                self.kernel = kernel
            else:
                raise QiskitError('Invalid kernel object is specified.')
        else:
            self.kernel = Kernel()

    def __eq__(self, other):
        """Two Acquires are the same if they are of the same type
        and have the same kernel and discriminator.

        Args:
            other (Acquire): Other Acquire

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self.kernel == other.kernel and \
                self.discriminator == other.discriminator:
            return True
        return False


class Discriminator(MeasOpts):
    """Discriminator."""

    def __init__(self, name=None, **params):
        """Create new discriminator.

        Parameters:
            name (str): Name of discriminator to be used.
        """
        super(Discriminator, self).__init__(name, **params)


class Kernel(MeasOpts):
    """Kernel."""

    def __init__(self, name=None, **params):
        """Create new kernel.

        Parameters:
            name (str): Name of kernel to be used.
        """
        super(Kernel, self).__init__(name, **params)
