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
from warnings import warn

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
        mixed_frame: Optional[Union[PulseChannel, MixedFrame]] = None,
        pulse_target: Optional[PulseTarget] = None,
        frame: Optional[Frame] = None,
        name: Optional[str] = None,
        channel: Optional[PulseChannel] = None,
    ):
        """Create a new pulse instruction.

        Args:
            pulse: A pulse waveform description, such as
                   :py:class:`~qiskit.pulse.library.Waveform`.
            mixed_frame: A ``MixedFrame`` on which the pulse is to be applied.
            pulse_target: A ``PulseTarget`` on which the pulse is to be applied. Has to be
                provided when frame is provided, and can not be provided if ``MixedFrame``
                was provided.
            frame: The ``Frame`` of the carrier wave. Has to be provided when pulse_target is
                provided, and can not be provided if ``MixedFrame`` was provided.
            channel: The channel to which the pulse is applied. Can not be provided
                if ``MixedFrame`` or ``Frame`` and ``PulseTarget`` were provided.
            name: Name of the instruction for display purposes. Defaults to ``pulse.name``.

        Raises:
            PulseError: If ``frame`` and ``pulse_target`` are not provided together.
            PulseError: If ``mixed_frame`` is provided together with ``frame`` and ``pulse_target``.
            PulseError: If ``channel`` is provided together with ``mixed_frame``
                or ``frame`` and ``pulse_target``.
        """
        if isinstance(mixed_frame, PulseChannel):
            msg = (
                "The signature of Play() has changed in Qiskit 0.46.0. "
                "Calling Play(my_pulse, my_channel"
                + (", my_name" if isinstance(pulse_target, str) else "")
                + ") "
                "is now deprecated. Use Play(my_pulse, channel = my_channel"
                + (", name=my_name" if isinstance(pulse_target, str) else "")
                + ") instead."
            )
            channel = mixed_frame
            mixed_frame = None
            if isinstance(pulse_target, str):
                name = pulse_target
                pulse_target = None

            warn(msg, DeprecationWarning, stacklevel=2)

        if (frame is None) != (pulse_target is None):
            raise PulseError("Both frame and pulse_target must be provided together.")

        if mixed_frame is not None and frame is not None:
            raise PulseError("Can't provide mixed_frame with frame and pulse_target.")

        if frame is not None:
            mixed_frame = MixedFrame(pulse_target=pulse_target, frame=frame)

        if channel is not None:
            warn(
                "Using PulseChannel for instructions is pending deprecation, as of Qiskit 0.46.0."
                "See qiskit.pulse.model for more details about the new model which includes"
                "PulseTarget and Frame.",
                PendingDeprecationWarning,
                stacklevel=2,
            )
            if mixed_frame is not None:
                raise PulseError(
                    "Can't provide channel with either mixed_frame or frame and pulse_target"
                )
            mixed_frame = channel

        if name is None:
            name = pulse.name
        super().__init__(operands=(pulse, mixed_frame), name=name)

    def _validate(self):
        """Called after initialization to validate instruction data.

        Raises:
            PulseError: If pulse is not a Pulse type.
            PulseError: If the input does not create a valid target for the instruction
                (``Channel`` or ``MixedFrame`` or ``PulseTarget`` and ``Frame``).
        """
        if not isinstance(self.pulse, Pulse):
            raise PulseError("The `pulse` argument to `Play` must be of type `library.Pulse`.")

        if not isinstance(self.mixed_frame, (PulseChannel, MixedFrame)):
            raise PulseError(
                f"Expected a valid target "
                f"(``Channel`` or ``MixedFrame`` or ``PulseTarget`` and ``Frame``) "
                f"instead got {self.operands[1]}"
            )

    @property
    def pulse(self) -> Pulse:
        """A description of the samples that will be played."""
        return self.operands[0]

    @property
    def channel(self) -> PulseChannel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on.

        If the instruction is not scheduled on a channel, return ``None``.
        """
        return self.operands[1] if isinstance(self.operands[1], PulseChannel) else None

    @property
    def mixed_frame(self) -> Union[MixedFrame, PulseChannel]:
        """Return the mixed_frame that this instruction is
        scheduled on.

        For backward compatibility the returned object could be of type :class:`.PulseChannel`.
        """
        return self.operands[1]

    @property
    def channels(self) -> Tuple[PulseChannel]:
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
