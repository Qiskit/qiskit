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

"""
Acquire.
"""
import warnings
from typing import Optional, Union, List

from qiskit.pulse.channels import Qubit, MemorySlot, RegisterSlot, AcquireChannel
from qiskit.pulse.exceptions import PulseError
from .instruction import Instruction
from .meas_opts import Discriminator, Kernel
from .command import Command


class Acquire(Command):
    """Acquire."""

    ALIAS = 'acquire'
    prefix = 'acq'

    def __init__(self, duration: int, discriminator: Optional[Discriminator] = None,
                 kernel: Optional[Kernel] = None, name: Optional[str] = None):
        """Create new acquire command.

        Args:
            duration: Duration of acquisition
            discriminator: Discriminators to be used (from the list of available discriminator)
                if the measurement level is 2
            kernel: The data structures defining the measurement kernels
                to be used (from the list of available kernels) and set of parameters
                (if applicable) if the measurement level is 1 or 2.
            name: Name of this command.

        Raises:
            PulseError: when invalid discriminator or kernel object is input.
        """
        super().__init__(duration=duration)

        self._name = Acquire.create_name(name)

        if discriminator:
            if isinstance(discriminator, Discriminator):
                self._discriminator = discriminator
            else:
                raise PulseError('Invalid discriminator object is specified.')
        else:
            self._discriminator = None

        if kernel:
            if isinstance(kernel, Kernel):
                self._kernel = kernel
            else:
                raise PulseError('Invalid kernel object is specified.')
        else:
            self._kernel = None

    @property
    def kernel(self):
        """Return kernel settings."""
        return self._kernel

    @property
    def discriminator(self):
        """Return discrimination settings."""
        return self._discriminator

    def __eq__(self, other: 'Acquire'):
        """Two Acquires are the same if they are of the same type
        and have the same kernel and discriminator.

        Args:
            other: Other Acquire

        Returns:
            bool: are self and other equal.
        """
        return (super().__eq__(other) and
                self.kernel == other.kernel and
                self.discriminator == other.discriminator)

    def __hash__(self):
        return hash((super().__hash__(), self.kernel, self.discriminator))

    def __repr__(self):
        return '%s(%s, duration=%d, kernel=%s, discriminator=%s)' % \
               (self.__class__.__name__, self.name, self.duration,
                self.kernel, self.discriminator)

    # pylint: disable=arguments-differ
    def to_instruction(self,
                       qubit: Union[AcquireChannel, List[AcquireChannel]],
                       mem_slot: Optional[MemorySlot] = None,
                       reg_slot: Optional[RegisterSlot] = None,
                       mem_slots: Optional[List[MemorySlot]] = None,
                       reg_slots: Optional[List[RegisterSlot]] = None,
                       name: Optional[str] = None) -> 'AcquireInstruction':

        return AcquireInstruction(self, qubit, mem_slot, reg_slot,
                                  mem_slots=mem_slots, reg_slots=reg_slots,
                                  name=name)
    # pylint: enable=arguments-differ


class AcquireInstruction(Instruction):
    """Pulse to acquire measurement result."""

    def __init__(self,
                 command: Acquire,
                 acquire: Union[AcquireChannel, List[AcquireChannel]],
                 mem_slot: Optional[MemorySlot] = None,
                 reg_slot: Optional[RegisterSlot] = None,
                 mem_slots: Optional[List[MemorySlot]] = None,
                 reg_slots: Optional[List[RegisterSlot]] = None,
                 name: Optional[str] = None):

        if isinstance(acquire, list) or mem_slots or reg_slots:
            warnings.warn('AcquireInstruction on multiple qubits, multiple memory slots '
                          'and multiple reg slots has been deprecated. Parameter "mem_slots" '
                          'has been replaced by "mem_slot" and "reg_slots" has been '
                          'replaced by "reg_slot"', DeprecationWarning, 3)

        if not isinstance(acquire, list):
            acquire = [acquire]

        if isinstance(acquire[0], Qubit):
            raise PulseError("AcquireInstruction can not be instantiated with Qubits, "
                             "which are deprecated.")

        if mem_slot:
            mem_slots = [mem_slot]

        if reg_slot:
            reg_slots = [reg_slot]

        if not (mem_slots or reg_slots):
            raise PulseError('Neither memoryslots or registers were supplied')

        if mem_slots:
            if len(acquire) != len(mem_slots):
                raise PulseError("#mem_slots must be equals to #acquires")

        if reg_slots:
            if len(acquire) != len(reg_slots):
                raise PulseError("#reg_slots must be equals to #acquires")
        else:
            reg_slots = []

        super().__init__(command, *acquire, *mem_slots, *reg_slots, name=name)

        self._acquires = acquire
        self._mem_slots = mem_slots
        self._reg_slots = reg_slots

    @property
    def acquire(self):
        """Acquire channel to be acquired on."""
        return self._acquires[0] if self._acquires else None

    @property
    def mem_slot(self):
        """MemorySlot."""
        return self._mem_slots[0] if self._mem_slots else None

    @property
    def reg_slot(self):
        """RegisterSlot."""
        return self._reg_slots[0] if self._reg_slots else None

    @property
    def acquires(self):
        """Acquire channels to be acquired on."""
        warnings.warn('"acquires" is deprecated and being replaced by "acquire"',
                      DeprecationWarning, 3)
        return self._acquires

    @property
    def mem_slots(self):
        """MemorySlots."""
        warnings.warn('"mem_slots" is deprecated and being replaced by "mem_slot"',
                      DeprecationWarning, 3)
        return self._mem_slots

    @property
    def reg_slots(self):
        """RegisterSlots."""
        warnings.warn('"reg_slots" is deprecated and being replaced by "reg_slot"',
                      DeprecationWarning, 3)
        return self._reg_slots
