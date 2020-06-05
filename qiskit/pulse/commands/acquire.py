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

from typing import Optional

from ..channels import MemorySlot, RegisterSlot, AcquireChannel
from ..exceptions import PulseError

# pylint: disable=unused-import

from ..instructions import Acquire
from ..instructions import Instruction


class AcquireInstruction(Instruction):
    """Deprecated."""

    def __init__(self,
                 command: Acquire,
                 acquire: AcquireChannel,
                 mem_slot: Optional[MemorySlot] = None,
                 reg_slot: Optional[RegisterSlot] = None,
                 name: Optional[str] = None):

        warnings.warn("The ``AcquireInstruction`` has been deprecated. Please use Acquire with "
                      "channels instead. For example, AcquireInstruction(Acquire(duration), "
                      "AcquireChannel(0), MemorySlot(0)) becomes Acquire(duration, "
                      "AcquireChannel(0), MemorySlot(0)).",
                      DeprecationWarning)

        if isinstance(acquire, list) or isinstance(mem_slot, list) or isinstance(reg_slot, list):
            raise PulseError("The Acquire instruction takes only one AcquireChannel and one "
                             "classical memory destination for the measurement result.")

        if not (mem_slot or reg_slot):
            raise PulseError('Neither memoryslots or registers were supplied')

        all_channels = [chan for chan in [acquire, mem_slot, reg_slot] if chan is not None]
        super().__init__((), command, all_channels, name=name)

        self._acquire = acquire
        self._mem_slot = mem_slot
        self._reg_slot = reg_slot

    @property
    def acquire(self):
        """Acquire channel to be acquired on."""
        return self._acquire

    @property
    def channel(self):
        """Acquire channel to be acquired on."""
        return self._acquire

    @property
    def mem_slot(self):
        """MemorySlot."""
        return self._mem_slot

    @property
    def reg_slot(self):
        """RegisterSlot."""
        return self._reg_slot

    @property
    def acquires(self):
        """Acquire channels to be acquired on."""
        warnings.warn("Acquire.acquires is deprecated. Use the channel attribute instead.",
                      DeprecationWarning)
        return [self._acquire]

    @property
    def mem_slots(self):
        """MemorySlots."""
        warnings.warn("Acquire.mem_slots is deprecated. Use the mem_slot attribute instead.",
                      DeprecationWarning)
        return [self._mem_slot]

    @property
    def reg_slots(self):
        """RegisterSlots."""
        warnings.warn("Acquire.reg_slots is deprecated. Use the reg_slot attribute instead.",
                      DeprecationWarning)
        return [self._reg_slot]
