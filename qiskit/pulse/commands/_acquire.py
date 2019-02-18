# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-param-doc

"""
Acquire.
"""

from qiskit.exceptions import QiskitError
from qiskit.pulse.commands import PulseCommand


class Acquire(PulseCommand):
    """Acquire."""

    def __init__(self, duration, memory_slot,
                 register_slot=None, discriminator=None, kernel=None):
        """Create new acquire command.

        Args:
            duration (int): Duration of acquisition.
            memory_slot (int): Index of the classical memory slots to store
                the measurement results. The total number of memory slots is
                specified through the Qobj memory slot numbering which starts at 0.
                Memory slots can be overwritten if they are specified twice
                in the same experiment (separate experiments have separate memories).
            register_slot (int): Index of the classical register slots in which to
                store the measurement results. The total number of register slots
                is specified by the backend. This is only allowed if the backend
                supports conditionals and the memory level is 2 (discriminated).
                Registers can only accept bits.
            discriminator (Discriminator): Discriminators to be used
                (from the list of available discriminator) if the measurement level is 2.
            kernel (Kernel): The data structures defining the measurement kernels
                to be used (from the list of available kernels) and set of parameters
                (if applicable) if the measurement level is 1 or 2.

        Raises:
            QiskitError: when invalid discriminator or kernel object is input.
        """

        super(Acquire, self).__init__(duration)

        self.memory_slot = memory_slot
        self.register_slot = register_slot

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


class Discriminator:
    """Discriminator."""

    def __init__(self, name=None, **params):
        """Create new discriminator.

        Parameters:
            name (str): Name of discriminator to be used.
        """
        self.name = name
        self.params = params


class Kernel:
    """Kernel."""

    def __init__(self, name=None, **params):
        """Create new kernel.

        Parameters:
            name (str): Name of kernel to be used.
        """
        self.name = name
        self.params = params
