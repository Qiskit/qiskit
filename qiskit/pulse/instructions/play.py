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
from typing import Dict, Optional, Union, Tuple, Any, Set

from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library.pulse import Pulse
from qiskit.pulse.library.signal import Signal
from qiskit.pulse.instructions.instruction import Instruction
from qiskit.pulse.utils import deprecated_functionality
from qiskit.pulse.frame import Frame


class Play(Instruction):
    """This instruction is responsible for applying a pulse on a channel.

    The pulse specifies the exact time dynamics of the output signal envelope for a limited
    time. The output is modulated by a phase and frequency which are controlled by separate
    instructions. The pulse duration must be fixed, and is implicitly given in terms of the
    cycle time, dt, of the backend.
    """

    def __init__(
        self, pulse: Union[Pulse, Signal], channel: PulseChannel, name: Optional[str] = None
    ):
        """Create a new pulse instruction.

        Args:
            pulse: A pulse waveform description, such as
                   :py:class:`~qiskit.pulse.library.Waveform`.
            channel: The channel to which the pulse is applied.
            name: Name of the instruction for display purposes. Defaults to ``pulse.name``.

        Raises:
            PulseError: If pulse is not a Pulse type.
        """
        if not isinstance(pulse, (Pulse, Signal)):
            raise PulseError(
                "The `pulse` argument to `Play` must be of type `library.Pulse` or "
                "`library.Signal`."
            )

        if not isinstance(channel, PulseChannel):
            raise PulseError(
                "The `channel` argument to `Play` must be of type `channels.PulseChannel`."
            )
        if name is None:
            name = pulse.name
        super().__init__(operands=(pulse, channel), name=name)

    @property
    def pulse(self) -> Pulse:
        """A description of the samples that will be played."""
        if isinstance(self.operands[0], Signal):
            return self.operands[0].pulse

        return self.operands[0]

    @property
    def signal(self) -> Signal:
        """The signal that will be played."""
        if isinstance(self.operands[0], Pulse):
            return Signal(self.operands[0], self.operands[1].frame)

        return self.operands[0]

    @property
    def frame(self) -> Frame:
        """The frame in which the pulse is played."""
        if isinstance(self.operands[0], Signal):
            return self.operands[0].frame

        return self.channel.frame

    @property
    def channel(self) -> PulseChannel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on.
        """
        return self.operands[1]

    @property
    def channels(self) -> Tuple[PulseChannel]:
        """Returns the channels that this schedule uses."""
        return (self.channel,)

    @property
    def frames(self) -> Tuple[Frame]:
        """Return the frames this instructions acts on."""
        return (self.frame,)

    @property
    def duration(self) -> Union[int, ParameterExpression]:
        """Duration of this instruction."""
        return self.pulse.duration

    def _initialize_parameter_table(self, operands: Tuple[Any]):
        """A helper method to initialize parameter table.

        Args:
            operands: List of operands associated with this instruction.
        """
        super()._initialize_parameter_table(operands)

        if any(isinstance(val, ParameterExpression) for val in self.pulse.parameters.values()):
            for value in self.pulse.parameters.values():
                if isinstance(value, ParameterExpression):
                    for param in value.parameters:
                        # Table maps parameter to operand index, 0 for ``pulse``
                        self._parameter_table[param].append(0)

    @property
    def parameters(self) -> Set:
        """Parameters which determine the instruction behavior."""
        parameters = set()
        for pulse_param_expr in self.pulse.parameters.values():
            if isinstance(pulse_param_expr, ParameterExpression):
                for pulse_param in pulse_param_expr.parameters:
                    parameters.add(pulse_param)
        if self.channel.is_parameterized():
            for ch_param in self.channel.parameters:
                parameters.add(ch_param)

        return parameters

    @deprecated_functionality
    def assign_parameters(
        self, value_dict: Dict[ParameterExpression, ParameterValueType]
    ) -> "Play":
        super().assign_parameters(value_dict)
        pulse = self.pulse.assign_parameters(value_dict)
        self._operands = (pulse, self.channel)
        return self

    def is_parameterized(self) -> bool:
        """Return True iff the instruction is parameterized."""
        return self.pulse.is_parameterized() or super().is_parameterized()
