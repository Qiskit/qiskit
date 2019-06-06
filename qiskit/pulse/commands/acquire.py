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
from typing import Union, List

from qiskit.pulse.channels import Qubit, MemorySlot, RegisterSlot, AcquireChannel
from qiskit.pulse.exceptions import PulseError
from .instruction import Instruction
from .meas_opts import Discriminator, Kernel
from .command import Command


class Acquire(Command):
    """Acquire."""

    ALIAS = 'acquire'

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
            PulseError: when invalid discriminator or kernel object is input.
        """
        super().__init__(duration=duration)

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

    def __repr__(self):
        return '%s(%s, duration=%d, kernel=%s, discriminator=%s)' % \
               (self.__class__.__name__, self.name, self.duration,
                self.kernel, self.discriminator)

    # pylint: disable=arguments-differ
    def to_instruction(self,
                       qubits: Union[Qubit, List[Qubit]],
                       mem_slots: Union[MemorySlot, List[MemorySlot]] = None,
                       reg_slots: Union[RegisterSlot, List[RegisterSlot]] = None,
                       name=None) -> 'AcquireInstruction':
        return AcquireInstruction(self, qubits, mem_slots, reg_slots, name=name)
    # pylint: enable=arguments-differ


class AcquireInstruction(Instruction):
    """Pulse to acquire measurement result. """

    def __init__(self,
                 command: Acquire,
                 qubits: Union[Qubit, AcquireChannel, List[Qubit], List[AcquireChannel]],
                 mem_slots: Union[MemorySlot, List[MemorySlot]],
                 reg_slots: Union[RegisterSlot, List[RegisterSlot]] = None,
                 name=None):

        if isinstance(qubits, (Qubit, AcquireChannel)):
            qubits = [qubits]

        if not (mem_slots or reg_slots):
            raise PulseError('Neither memoryslots or registers were supplied')

        if mem_slots:
            if isinstance(mem_slots, MemorySlot):
                mem_slots = [mem_slots]
            elif len(qubits) != len(mem_slots):
                raise PulseError("#mem_slots must be equals to #qubits")

        if reg_slots:
            if isinstance(reg_slots, RegisterSlot):
                reg_slots = [reg_slots]
            if len(qubits) != len(reg_slots):
                raise PulseError("#reg_slots must be equals to #qubits")
        else:
            reg_slots = []

        # extract acquire channels
        acquires = []
        for q in qubits:
            if isinstance(q, Qubit):
                q = q.acquire
            acquires.append(q)

        super().__init__(command, *acquires, *mem_slots, *reg_slots, name=name)

        self._acquires = acquires
        self._mem_slots = mem_slots
        self._reg_slots = reg_slots

    @property
    def acquires(self):
        """Acquire channels to be acquired on. """
        return self._acquires

    @property
    def mem_slots(self):
        """MemorySlots. """
        return self._mem_slots

    @property
    def reg_slots(self):
        """RegisterSlots. """
        return self._reg_slots
