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

"""An instruction to transmit a given pulse on a ``PulseChannel`` (i.e., those which support
transmitted pulses, such as ``DriveChannel``).
"""
from typing import Optional, Union, Tuple, Set

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.instructions.instruction import Instruction
from qiskit.pulse.library.pulse import Pulse
from qiskit.pulse.model import PulseTarget, Frame, MixedFrame


class Play(Instruction):
    """This instruction is responsible for applying a pulse on a channel.

    The pulse specifies the exact time dynamics of the output signal envelope for a limited
    time. The output is modulated by a phase and frequency which are controlled by separate
    instructions. The pulse duration must be fixed, and is implicitly given in terms of the
    cycle time, dt, of the backend.
    """

    def __init__(
        self,
        pulse: Pulse,
        *,
        target: Optional[PulseTarget] = None,
        frame: Optional[Frame] = None,
        mixed_frame: Optional[MixedFrame] = None,
        channel: Optional[PulseChannel] = None,
        name: Optional[str] = None,
    ):
        """Create a new pulse play instruction.

        To uniquely identify the mixed frame to which the pulse is applied, the arguments must include
        exactly one of ``mixed_frame``, ``channel`` or the duo ``target`` and ``frame``.

        Args:
            pulse: A pulse waveform description, such as
                   :py:class:`~qiskit.pulse.library.Waveform`.
            target: The target to which the pulse is applied.
            frame: The frame characterizing the pulse carrier.
            mixed_frame: The mixed frame to which the pulse is applied.
            channel: The channel to which the pulse is applied.
            name: Name of the instruction for display purposes. Defaults to ``pulse.name``.

        Raises:
            PulseError: If the mixed frame is under or over specified.
        """
        if (target is not None) != (frame is not None):
            raise PulseError("target and frame must be provided simultaneously")
        if (channel is not None) + (mixed_frame is not None) + (target is not None) != 1:
            raise PulseError(
                "Exactly one of mixed_frame, channel or the duo target and frame has to be provided"
            )

        if target is not None:
            mixed_frame = MixedFrame(target, frame)
        else:
            mixed_frame = mixed_frame or channel

        if name is None:
            name = pulse.name
        super().__init__(operands=(pulse, mixed_frame), name=name)

    def _validate(self):
        """Called after initialization to validate instruction data.

        Raises:
            PulseError: If pulse is not a Pulse type.
            PulseError: If the input ``channel`` is not type :class:`PulseChannel`.
        """
        if not isinstance(self.pulse, Pulse):
            raise PulseError("The `pulse` argument to `Play` must be of type `library.Pulse`.")

        if not isinstance(self.mixed_frame, (PulseChannel, MixedFrame)):
            raise PulseError(
                f"Expected a mixed frame or pulse channel, got {self.mixed_frame} instead."
            )

    @property
    def pulse(self) -> Pulse:
        """A description of the samples that will be played."""
        return self.operands[0]

    @property
    def channel(self) -> Union[PulseChannel, None]:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on.
        """
        if isinstance(self.operands[1], PulseChannel):
            return self.operands[1]
        return None

    @property
    def mixed_frame(self) -> Union[PulseChannel, MixedFrame]:
        """Return the mixed frame that this instruction is
        scheduled on.
        """
        return self.operands[1]

    @property
    def channels(self) -> Tuple[Union[PulseChannel, None]]:
        """Returns the channels that this schedule uses."""
        return (self.channel,)

    @property
    def duration(self) -> Union[int, ParameterExpression]:
        """Duration of this instruction."""
        return self.pulse.duration

    @property
    def parameters(self) -> Set:
        """Parameters which determine the instruction behavior."""
        parameters = set()

        # Note that Pulse.parameters returns dict rather than set for convention.
        # We need special handling for Play instruction.
        for pulse_param_expr in self.pulse.parameters.values():
            if isinstance(pulse_param_expr, ParameterExpression):
                parameters = parameters | pulse_param_expr.parameters

        if self.channel is not None and self.channel.is_parameterized():
            parameters = parameters | self.channel.parameters

        return parameters
