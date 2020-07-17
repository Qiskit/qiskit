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

"""The Acquire instruction is used to trigger the qubit measurement unit and provide
some metadata for the acquisition process, for example, where to store classified readout data.
"""
import warnings

from typing import List, Optional

from ..channels import MemorySlot, RegisterSlot, AcquireChannel
from ..configuration import Kernel, Discriminator
from ..exceptions import PulseError
from .instruction import Instruction


class Acquire(Instruction):
    """The Acquire instruction is used to trigger the ADC associated with a particular qubit;
    e.g. instantiated with AcquireChannel(0), the Acquire command will trigger data collection
    for the channel associated with qubit 0 readout. This instruction also provides acquisition
    metadata:

     * the number of cycles during which to acquire (in terms of dt),

     * the register slot to store classified, intermediary readout results,

     * the memory slot to return classified results,

     * the kernel to integrate raw data for each shot, and

     * the discriminator to classify kerneled IQ points.
    """

    def __init__(self,
                 duration: int,
                 channel: AcquireChannel,
                 mem_slot: Optional[MemorySlot] = None,
                 reg_slot: Optional[RegisterSlot] = None,
                 kernel: Optional[Kernel] = None,
                 discriminator: Optional[Discriminator] = None,
                 name: Optional[str] = None):
        """Create a new Acquire instruction.

        Args:
            duration: Length of time to acquire data in terms of dt.
            channel: The channel that will acquire data.
            mem_slot: The classical memory slot in which to store the classified readout result.
            reg_slot: The fast-access register slot in which to store the classified readout
                      result for fast feedback.
            kernel: A ``Kernel`` for integrating raw data.
            discriminator: A ``Discriminator`` for discriminating kerneled IQ data into 0/1
                           results.
            name: Name of the instruction for display purposes.

        Raises:
            PulseError: If channels are supplied, and the number of register and/or memory slots
                        does not equal the number of channels.
        """
        if isinstance(channel, list) or isinstance(mem_slot, list) or isinstance(reg_slot, list):
            raise PulseError("The Acquire instruction takes only one AcquireChannel and one "
                             "classical memory destination for the measurement result.")

        if not (mem_slot or reg_slot):
            raise PulseError('Neither MemorySlots nor RegisterSlots were supplied.')

        self._channel = channel
        self._mem_slot = mem_slot
        self._reg_slot = reg_slot
        self._kernel = kernel
        self._discriminator = discriminator

        all_channels = [chan for chan in [channel, mem_slot, reg_slot] if chan is not None]
        super().__init__((duration, self.channel, self.mem_slot, self.reg_slot),
                         duration, all_channels, name=name)

    @property
    def channel(self) -> AcquireChannel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on.
        """
        return self._channel

    @property
    def kernel(self) -> Kernel:
        """Return kernel settings."""
        return self._kernel

    @property
    def discriminator(self) -> Discriminator:
        """Return discrimination settings."""
        return self._discriminator

    @property
    def acquire(self) -> AcquireChannel:
        """Acquire channel to acquire data. The ``AcquireChannel`` index maps trivially to
        qubit index.
        """
        return self._channel

    @property
    def mem_slot(self) -> MemorySlot:
        """The classical memory slot which will store the classified readout result."""
        return self._mem_slot

    @property
    def reg_slot(self) -> RegisterSlot:
        """The fast-access register slot which will store the classified readout result for
        fast-feedback computation.
        """
        return self._reg_slot

    @property
    def acquires(self) -> List[AcquireChannel]:
        """Acquire channels to be acquired on."""
        warnings.warn("Acquire.acquires is deprecated. Use the channel attribute instead.",
                      DeprecationWarning)
        return [self._channel]

    @property
    def mem_slots(self) -> List[MemorySlot]:
        """MemorySlots."""
        warnings.warn("Acquire.mem_slots is deprecated. Use the mem_slot attribute instead.",
                      DeprecationWarning)
        return [self._mem_slot]

    @property
    def reg_slots(self) -> List[RegisterSlot]:
        """RegisterSlots."""
        warnings.warn("Acquire.reg_slots is deprecated. Use the reg_slot attribute instead.",
                      DeprecationWarning)
        return [self._reg_slot]

    def __repr__(self) -> str:
        return "{}({}{}{}{}{}{})".format(
            self.__class__.__name__,
            self.duration,
            ', ' + str(self.channel),
            ', ' + str(self.mem_slot) if self.mem_slot else '',
            ', ' + str(self.reg_slot) if self.reg_slot else '',
            ', ' + str(self.kernel) if self.kernel else '',
            ', ' + str(self.discriminator) if self.discriminator else '')
