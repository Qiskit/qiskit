# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""TODO"""
import warnings

from typing import List, Optional, Union

from ..channels import MemorySlot, RegisterSlot, AcquireChannel
from ..configuration import Kernel, Discriminator
from ..exceptions import PulseError
from .instruction import Instruction


class Acquire(Instruction):
    """TODO"""

    def __init__(self,
                 duration: int,
                 channel: Optional[Union[AcquireChannel, List[AcquireChannel]]] = None,
                 mem_slot: Optional[Union[MemorySlot, List[MemorySlot]]] = None,
                 reg_slots: Optional[Union[RegisterSlot, List[RegisterSlot]]] = None,
                 mem_slots: Optional[Union[List[MemorySlot]]] = None,
                 reg_slot: Optional[RegisterSlot] = None,
                 kernel: Optional[Kernel] = None,
                 discriminator: Optional[Discriminator] = None,
                 name: Optional[str] = None):
        """"""

        if isinstance(acquire, list) or isinstance(mem_slot, list) or reg_slots:
            warnings.warn('The AcquireInstruction on multiple qubits, multiple '
                          'memory slots and multiple reg slots is deprecated. The '
                          'parameter "mem_slots" has been replaced by "mem_slot" and '
                          '"reg_slots" has been replaced by "reg_slot"', DeprecationWarning, 3)

        if not isinstance(acquire, list):
            acquire = [acquire]

        if mem_slot and not isinstance(mem_slot, list):
            mem_slot = [mem_slot]
        elif mem_slots:
            mem_slot = mem_slots

        if reg_slot:
            reg_slot = [reg_slot]
        elif reg_slots and not isinstance(reg_slots, list):
            reg_slot = [reg_slots]
        else:
            reg_slot = reg_slots

        if not (mem_slot or reg_slot):
            raise PulseError('Neither memoryslots or registers were supplied')

        if mem_slot and len(acquire) != len(mem_slot):
            raise PulseError("The number of mem_slots must be equals to the number of acquires")

        if reg_slot:
            if len(acquire) != len(reg_slot):
                raise PulseError("The number of reg_slots must be equals "
                                 "to the number of acquires")
        else:
            reg_slot = []

        super().__init__(duration, *acquire, *mem_slot, *reg_slot, name=name)

        self._acquires = acquire
        self._mem_slots = mem_slot
        self._reg_slots = reg_slot
        self._kernel = kernel
        self._discriminator = discriminator

    @property
    def channel(self):
        """TODO"""
        return self._channel

    @property
    def operands(self):
        """TODO"""
        return [self.duration, self.channel,
                self.mem_slot, self.reg_slot,
                self.kernel, self.discriminator]

    @property
    def kernel(self):
        """Return kernel settings."""
        return self._kernel

    @property
    def discriminator(self):
        """Return discrimination settings."""
        return self._discriminator

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
        return self._acquires

    @property
    def mem_slots(self):
        """MemorySlots."""
        return self._mem_slots

    @property
    def reg_slots(self):
        """RegisterSlots."""
        return self._reg_slots

    def __call__(self,
                 channel: Optional[Union[AcquireChannel, List[AcquireChannel]]] = None,
                 mem_slot: Optional[Union[MemorySlot, List[MemorySlot]]] = None,
                 reg_slots: Optional[Union[RegisterSlot, List[RegisterSlot]]] = None,
                 mem_slots: Optional[Union[List[MemorySlot]]] = None,
                 reg_slot: Optional[RegisterSlot] = None,
                 kernel: Optional[Kernel] = None,
                 discriminator: Optional[Discriminator] = None) -> 'Acquire':
        """Return new ``Acquire`` that is fully instantiated with its channels.

        Args:
            channel: The channel that will have the delay.

        Return:
            Complete and ready to schedule ``Delay``.

        Raises:
            PulseError: If ``channel`` has already been set.
        """
        warnings.warn("Calling Acquire with a channel is deprecated. Instantiate the acquire with "
                      "a channel instead.", DeprecationWarning)
        if self._channel is not None:
            raise PulseError("The channel has already been assigned as {}.".format(self.channel))
        return Acquire(self.duration,
                       channel=channel,
                       mem_slot=mem_slot,
                       reg_slots=reg_slots,
                       mem_slots=mem_slots,
                       reg_slot=reg_slot,
                       kernel=kernel,
                       discriminator=discriminator)
