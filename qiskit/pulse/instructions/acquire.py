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

"""The Acquire instruction is used to trigger the analog-to-digital converter (ADC) and provide
some metadata for the acquisition process; for example, where to store classified readout data.
"""
import warnings

from typing import List, Optional, Union

from ..channels import MemorySlot, RegisterSlot, AcquireChannel
from ..configuration import Kernel, Discriminator
from ..exceptions import PulseError
from .instruction import Instruction


class Acquire(Instruction):
    """The Acquire instruction is used to trigger the ADC associated with a particular qubit;
    e.g. instantiated with AcquireChannel(0), the Acquire command will trigger data collection
    for the channel associated with qubit 0 readout. This instruction also provides acquisition
    metadata:
     - the duration of time to acquire data (in number of timesteps, dt),
     - the register slot to store classified, intermediary readout results,
     - the memory slot to return classified results,
     - the kernel to integrate raw data for each shot, and
     - the discriminator to classify kerneled IQ points.
    """

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
        """Create a new Acquire instruction.

        Args:
            duration: Length of time to acquire data in terms of dt.
            channel: The channel that will acquire data.
            mem_slot: The classical memory slot in which to store the classified readout result.
            mem_slots: Deprecated list form of ``mem_slot``.
            reg_slots: Deprecated list form of ``reg_slot``.
            reg_slot: The fast-access register slot in which to store the classified readout
                      result for fast feedback.
            kernel: A ``Kernel`` for integrating raw data.
            discriminator: A ``Discriminator`` for discriminating kerneled IQ data into 0/1
                           results.
            name: Name of the instruction for display purposes.

        Raises:
            PulseError: If the number of register and memory slots does not equal the number of
                        channels, or if channels are supplied and do not equal the number of m
                        does not equal the number of AcquireChannels supplied.
        """
        if isinstance(channel, list) or isinstance(mem_slot, list) or reg_slots or mem_slots:
            warnings.warn('The AcquireInstruction on multiple qubits, multiple '
                          'memory slots and multiple reg slots is deprecated. The '
                          'parameter "mem_slots" has been replaced by "mem_slot" and '
                          '"reg_slots" has been replaced by "reg_slot"', DeprecationWarning, 3)

        if not isinstance(channel, list):
            channels = [channel] if channel else None
        else:
            channels = channel

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

        if channels and not (mem_slot or reg_slot):
            raise PulseError('Neither MemorySlots nor RegisterSlots were supplied.')

        if channels and mem_slot and len(channels) != len(mem_slot):
            raise PulseError("The number of mem_slots must be equal to the number of channels.")

        if channels and reg_slot:
            if len(channels) != len(reg_slot):
                raise PulseError("The number of reg_slots must be equal to the number of "
                                 "channels.")
        else:
            reg_slot = []

        if name is None and channels is None:
            name = 'acq{:10x}'.format(hash((duration, kernel, discriminator)))
        elif name is None:
            name = 'acq{:10x}'.format(hash((duration, tuple(channels), tuple(mem_slot),
                                            tuple(reg_slot), kernel, discriminator)))

        if channels is not None:
            super().__init__(duration, *channels, *mem_slot, *reg_slot, name=name)
        else:
            super().__init__(duration, name=name)

        self._acquires = channels
        self._channel = channels[0] if channels else None
        self._mem_slots = mem_slot
        self._reg_slots = reg_slot
        self._kernel = kernel
        self._discriminator = discriminator

    @property
    def operands(self) -> List:
        """Return a list of instruction operands."""
        return [self.duration, self.channel,
                self.mem_slot, self.reg_slot,
                self.kernel, self.discriminator]

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
        return self._acquires[0] if self._acquires else None

    @property
    def mem_slot(self) -> MemorySlot:
        """The classical memory slot which will store the classified readout result."""
        return self._mem_slots[0] if self._mem_slots else None

    @property
    def reg_slot(self) -> RegisterSlot:
        """The fast-access register slot which will store the classified readout result for
        fast-feedback computation.
        """
        return self._reg_slots[0] if self._reg_slots else None

    @property
    def acquires(self) -> List[AcquireChannel]:
        """Acquire channels to be acquired on."""
        return self._acquires

    @property
    def mem_slots(self) -> List[MemorySlot]:
        """MemorySlots."""
        return self._mem_slots

    @property
    def reg_slots(self) -> List[RegisterSlot]:
        """RegisterSlots."""
        return self._reg_slots

    def __repr__(self):
        return "{}({}{}{}{}{}{})".format(
            self.__class__.__name__,
            self.duration,
            ', ' + ', '.join(str(ch) for ch in self.acquires) if self.acquires else '',
            ', ' + str(self.mem_slot) if self.mem_slot else '',
            ', ' + str(self.reg_slot) if self.reg_slot else '',
            ', ' + str(self.kernel) if self.kernel else '',
            ', ' + str(self.discriminator) if self.discriminator else '')

    def __eq__(self, other):
        return self.operands == other.operands

    def __call__(self,
                 channel: Optional[Union[AcquireChannel, List[AcquireChannel]]] = None,
                 mem_slot: Optional[Union[MemorySlot, List[MemorySlot]]] = None,
                 reg_slots: Optional[Union[RegisterSlot, List[RegisterSlot]]] = None,
                 mem_slots: Optional[Union[List[MemorySlot]]] = None,
                 reg_slot: Optional[RegisterSlot] = None,
                 kernel: Optional[Kernel] = None,
                 discriminator: Optional[Discriminator] = None,
                 name: Optional[str] = None) -> 'Acquire':
        """Return new ``Acquire`` that is fully instantiated with its channels.

        Args:
            channel: The channel that will acquire data.
            mem_slot: The classical memory slot in which to store the classified readout result.
            mem_slots: Deprecated list form of ``mem_slot``.
            reg_slots: Deprecated list form of ``reg_slot``.
            reg_slot: The fast-access register slot in which to store the classified readout
                      result for fast feedback.
            kernel: A ``Kernel`` for integrating raw data.
            discriminator: A ``Discriminator`` for discriminating kerneled IQ data into 0/1
                           results.
            name: Name of the instruction for display purposes.

        Return:
            Complete and ready to schedule ``Acquire``.

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
                       discriminator=discriminator,
                       name=name if name is not None else self.name)
