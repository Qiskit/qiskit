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

"""An instruction for blocking time on a channel; useful for scheduling alignment."""
from __future__ import annotations

from qiskit.circuit import ParameterExpression
from qiskit.pulse.channels import Channel
from qiskit.pulse.instructions.instruction import Instruction
from qiskit.pulse.model import Frame, PulseTarget, MixedFrame
from qiskit.pulse.exceptions import PulseError


class Delay(Instruction):
    """A blocking instruction with no other effect. The delay is used for aligning and scheduling
    other instructions.

    Example:

        To schedule an instruction at time = 10, on a channel assigned to the variable ``channel``,
        the following could be used::

            sched = Schedule(name="Delay instruction example")
            sched += Delay(10, channel)
            sched += Gaussian(duration, amp, sigma, channel)

        The ``channel`` will output no signal from time=0 up until time=10.
    """

    def __init__(
        self,
        duration: int | ParameterExpression,
        *,
        target: PulseTarget | None = None,
        frame: Frame | None = None,
        mixed_frame: MixedFrame | None = None,
        channel: Channel | None = None,
        name: str | None = None,
    ):
        """Create a new delay instruction.

        No other instruction may be scheduled within a ``Delay``.

        The delay can be set on a ``MixedFrame`` (=``Channel``) or a ``PulseTarget``. For the latter,
        provide only the ``target`` argument, and the delay will be broadcasted to all ``MixedFrame``s
        involving the ``PulseTarget``. For the former, provide exactly one of ``mixed_frame``,
        ``channel`` or the duo ``target`` and ``frame``, and the delay will apply only to the
        specified ``MixedFrame``.

        Args:
            duration: Length of time of the delay in terms of dt.
            target: The target that will have the delay.
            frame: The frame which in conjunction with ``target`` defines the mixed frame that will
                have the delay.
            mixed_frame: The mixed_frame that will have the delay.
            channel: The channel that will have the delay.
            name: Name of the delay for display purposes.
        Raises:
            PulseError: If the combination of ``target``, ``frame``, ``mixed_frame`` and ``channel``
                doesn't specify a unique ``MixedFrame`` or ``PulseTarget``.
            PulseError: If the inputs to ``target``, ``frame``, ``mixed_frame`` and ``channel``
                are not of the appropriate type.
        """
        if (channel is not None) + (mixed_frame is not None) + (target is not None) != 1:
            raise PulseError("Exactly one of mixed_frame, channel or target must be provided")

        if frame is not None:
            if target is None:
                raise PulseError("frame must be provided with target")
            inst_target = MixedFrame(target, frame)
        else:
            inst_target = mixed_frame or channel or target

        if not isinstance(inst_target, (Channel, MixedFrame, PulseTarget)):
            raise PulseError(
                f"Expected a MixedFrame, Channel or PulseTarget, got {inst_target} instead."
            )

        super().__init__(operands=(duration, inst_target), name=name)

    @property
    def inst_target(self) -> MixedFrame | PulseTarget | Channel:
        """Return the object targeted by this delay instruction."""
        return self.operands[1]

    @property
    def channel(self) -> Channel | None:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on.
        """
        if isinstance(self.operands[1], Channel):
            return self.operands[1]
        return None

    @property
    def channels(self) -> tuple[Channel]:
        """Returns the channels that this schedule uses."""
        return (self.channel,)

    @property
    def duration(self) -> int | ParameterExpression:
        """Duration of this instruction."""
        return self.operands[0]
