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

"""Acquire. Deprecated path."""
import warnings

from typing import Optional, Union, List

from ..channels import MemorySlot, RegisterSlot, AcquireChannel
from ..exceptions import PulseError

# pylint: disable=unused-import

from ..instructions import Acquire


class AcquireInstruction:
    """Deprecated."""

    def __init__(self,
                 command: Acquire,
                 acquire: Union[AcquireChannel, List[AcquireChannel]],
                 mem_slot: Optional[Union[MemorySlot, List[MemorySlot]]] = None,
                 reg_slots: Optional[Union[RegisterSlot, List[RegisterSlot]]] = None,
                 mem_slots: Optional[Union[List[MemorySlot]]] = None,
                 reg_slot: Optional[RegisterSlot] = None,
                 name: Optional[str] = None):

        warnings.warn("TODO", DeprecationWarning)

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

        super().__init__(command, *acquire, *mem_slot, *reg_slot, name=name)

        self._acquires = acquire
        self._mem_slots = mem_slot
        self._reg_slots = reg_slot

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
