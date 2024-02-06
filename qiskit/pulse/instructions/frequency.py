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

"""Frequency instructions module. These instructions allow the user to manipulate
the frequency of a channel.
"""
from typing import Optional, Union

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.instructions.instruction import FrameInstruction
from qiskit.pulse.model import Frame, MixedFrame, PulseTarget


class SetFrequency(FrameInstruction):
    r"""The set frequency instruction sets the modulation frequency of proceeding pulses
    associated with the frame the instruction is acting upon.

    In particular, played pulses take the form:

    .. math::
        Re[\exp(i 2\pi f jdt + \phi) d_j].

    Here, :math:`f` is the frequency of the modulation. The instruction ``SetFrequency`` allows
    the user to set the value of :math:`f`.

    The duration of ``SetFrequency`` is 0.
    """

    def __init__(
        self,
        frequency: Union[float, ParameterExpression],
        *,
        frame: Frame = None,
        target: PulseTarget = None,
        mixed_frame: MixedFrame = None,
        channel: PulseChannel = None,
        name: Optional[str] = None,
    ):
        """Creates a new set frequency instruction.

        The instruction can be set on a ``MixedFrame`` (=``Channel``) or a ``Frame``. For the latter,
        provide only the ``frame`` argument, and the instruction will be broadcasted to all
        ``MixedFrame``s involving the ``Frame``. For the former, provide exactly one of
        ``mixed_frame``, ``channel`` or the duo ``target`` and ``frame``, and the instruction
        will apply only to the specified ``MixedFrame``.

        Args:
            frequency: New frequency of the channel in Hz.
            frame: The frame the instruction will apply to.
            target: The target which in conjunction with ``frame`` defines the mixed frame that
                instruction will apply to.
            mixed_frame: The mixed_frame the instruction will apply to.
            channel: The channel the instruction will apply to.
            name: Name of this set frequency instruction.
        """
        inst_target = self._validate_and_format_frame(target, frame, mixed_frame, channel)
        super().__init__(operands=(frequency, inst_target), name=name)

    @property
    def frequency(self) -> Union[float, ParameterExpression]:
        """New frequency."""
        return self.operands[0]


class ShiftFrequency(FrameInstruction):
    r"""The shift frequency instruction updates the modulation frequency of proceeding pulses
    associated with the frame the instruction is acting upon.

    In particular, played pulses take the form:

    .. math::
        Re[\exp(i 2\pi f jdt + \phi) d_j].

    Here, :math:`f` is the frequency of the modulation. The instruction ``shiftFrequency`` allows
    the user to shift the value of :math:`f` by the value specified by the instructions.

    The duration of ``ShiftFrequency`` is 0.
    """

    def __init__(
        self,
        frequency: Union[float, ParameterExpression],
        *,
        frame: Frame = None,
        target: PulseTarget = None,
        mixed_frame: MixedFrame = None,
        channel: PulseChannel = None,
        name: Optional[str] = None,
    ):
        """Creates a new shift frequency instruction.

        The instruction can be set on a ``MixedFrame`` (=``Channel``) or a ``Frame``. For the latter,
        provide only the ``frame`` argument, and the instruction will be broadcasted to all
        ``MixedFrame``s involving the ``Frame``. For the former, provide exactly one of
        ``mixed_frame``, ``channel`` or the duo ``target`` and ``frame``, and the instruction
        will apply only to the specified ``MixedFrame``.

        Args:
            frequency: Frequency shift of the channel in Hz.
            frame: The frame the instruction will apply to.
            target: The target which in conjunction with ``frame`` defines the mixed frame that
                instruction will apply to.
            mixed_frame: The mixed_frame the instruction will apply to.
            channel: The channel the instruction will apply to.
            name: Name of this set frequency instruction.
        """
        inst_target = self._validate_and_format_frame(target, frame, mixed_frame, channel)
        super().__init__(operands=(frequency, inst_target), name=name)

    @property
    def frequency(self) -> Union[float, ParameterExpression]:
        """Frequency shift from the set frequency."""
        return self.operands[0]
