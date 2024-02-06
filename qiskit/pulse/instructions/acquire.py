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
from __future__ import annotations
from typing import Optional, Union
from qiskit.circuit import ParameterExpression
from qiskit.pulse.channels import MemorySlot, RegisterSlot, AcquireChannel
from qiskit.pulse.configuration import Kernel, Discriminator
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.instructions.instruction import Instruction
from qiskit.pulse.model import Qubit, Port


class Acquire(Instruction):
    """The Acquire instruction is used to trigger the ADC associated with a particular qubit.
    This instruction also provides acquisition metadata:

     * the number of cycles during which to acquire (in terms of dt),

     * the register slot to store classified, intermediary readout results,

     * the memory slot to return classified results,

     * the kernel to integrate raw data for each shot, and

     * the discriminator to classify kerneled IQ points.
    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        *,
        qubit: Optional[Qubit] = None,
        channel: Optional[AcquireChannel] = None,
        mem_slot: Optional[MemorySlot] = None,
        reg_slot: Optional[RegisterSlot] = None,
        kernel: Optional[Kernel] = None,
        discriminator: Discriminator | None = None,
        name: str | None = None,
    ):
        """Create a new Acquire instruction.

        The instruction can be instantiated either with respect to :class:`~.qiskit.pulse.Qubit`
        logical element, or :class:`AcquireChannel`.

        Args:
            duration: Length of time to acquire data in terms of dt.
            qubit: The qubit that will be acquired.
            channel: The channel that will acquire data.
            mem_slot: The classical memory slot in which to store the classified readout result.
            reg_slot: The fast-access register slot in which to store the classified readout
                      result for fast feedback.
            kernel: A ``Kernel`` for integrating raw data.
            discriminator: A ``Discriminator`` for discriminating kerneled IQ data into 0/1
                           results.
            name: Name of the instruction for display purposes.

        Raises:
            PulseError: If the input ``qubit`` is not type :class:`~.qiskit.pulse.Qubit`.
            PulseError: If the input ``channel`` is not type :class:`AcquireChannel`.
            PulseError: If the input ``mem_slot`` is not type :class:`MemorySlot`.
            PulseError: If the input ``reg_slot`` is not type :class:`RegisterSlot`.
            PulseError: When memory slot and register slot are both empty.
            PulseError: When both or none of ``qubit`` and ``channel`` are provided.
        """
        if qubit is not None and not isinstance(qubit, Qubit):
            raise PulseError(f"Expected a qubit, got {qubit} instead.")

        if channel is not None and not isinstance(channel, AcquireChannel):
            raise PulseError(f"Expected an acquire channel, got {channel} instead.")

        if (qubit is not None) + (channel is not None) != 1:
            raise PulseError("Expected exactly one of acq_element and channel.")

        if mem_slot and not isinstance(mem_slot, MemorySlot):
            raise PulseError(f"Expected a memory slot, got {mem_slot} instead.")

        if reg_slot and not isinstance(reg_slot, RegisterSlot):
            raise PulseError(f"Expected a register slot, got {reg_slot} instead.")

        if mem_slot is None and reg_slot is None:
            raise PulseError("Neither MemorySlots nor RegisterSlots were supplied.")

        super().__init__(
            operands=(duration, channel or qubit, mem_slot, reg_slot, kernel, discriminator),
            name=name,
        )

    @property
    def channel(self) -> Union[AcquireChannel, None]:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on.
        """
        if isinstance(self.operands[1], AcquireChannel):
            return self.operands[1]
        return None

    @property
    def qubit(self) -> Union[AcquireChannel, Qubit]:
        """Return the element acquired by this instruction."""
        return self.operands[1]

    @property
    def channels(self) -> tuple[AcquireChannel | MemorySlot | RegisterSlot, ...]:
        """Returns the channels that this schedule uses."""
        return tuple(self.operands[ind] for ind in (1, 2, 3) if self.operands[ind] is not None)

    @property
    def duration(self) -> int | ParameterExpression:
        """Duration of this instruction."""
        return self.operands[0]

    @property
    def kernel(self) -> Kernel:
        """Return kernel settings."""
        return self._operands[4]

    @property
    def discriminator(self) -> Discriminator:
        """Return discrimination settings."""
        return self._operands[5]

    @property
    def acquire(self) -> Union[AcquireChannel, Qubit, Port]:
        """Acquire channel to acquire data. The ``AcquireChannel`` index maps trivially to
        qubit index.
        """
        return self.operands[1]

    @property
    def mem_slot(self) -> MemorySlot:
        """The classical memory slot which will store the classified readout result."""
        return self.operands[2]

    @property
    def reg_slot(self) -> RegisterSlot:
        """The fast-access register slot which will store the classified readout result for
        fast-feedback computation.
        """
        return self.operands[3]

    def is_parameterized(self) -> bool:
        """Return True iff the instruction is parameterized."""
        return isinstance(self.duration, ParameterExpression) or super().is_parameterized()

    def __repr__(self) -> str:
        return "{}({}{}{}{}{}{})".format(
            self.__class__.__name__,
            self.duration,
            ", " + str(self.qubit),
            ", " + str(self.mem_slot) if self.mem_slot else "",
            ", " + str(self.reg_slot) if self.reg_slot else "",
            ", " + str(self.kernel) if self.kernel else "",
            ", " + str(self.discriminator) if self.discriminator else "",
        )
