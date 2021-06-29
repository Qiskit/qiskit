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
from typing import Optional, Union, Tuple
import warnings

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.channels import PulseChannel, DriveChannel, MeasureChannel, ControlChannel
from qiskit.pulse.frame import Frame
from qiskit.pulse.instructions.instruction import Instruction


class SetFrequency(Instruction):
    r"""Set the channel frequency. This instruction operates on ``PulseChannel`` s.
    A ``PulseChannel`` creates pulses of the form

    .. math::
        Re[\exp(i 2\pi f jdt + \phi) d_j].

    Here, :math:`f` is the frequency of the channel. The instruction ``SetFrequency`` allows
    the user to set the value of :math:`f`. All pulses that are played on a channel
    after SetFrequency has been called will have the corresponding frequency.

    The duration of SetFrequency is 0.
    """

    def __init__(
        self,
        frequency: Union[float, ParameterExpression],
        frame: Union[PulseChannel, Frame],
        name: Optional[str] = None,
    ):
        """Creates a new set channel frequency instruction.

        Args:
            frequency: New frequency of the channel in Hz.
            channel: The channel or frame this instruction operates on.
            name: Name of this set channel frequency instruction.
        """
        if isinstance(frame, PulseChannel):
            warnings.warn(
                f"Applying {self.__class__.__name__} to channel {frame}. This "
                f"functionality will be deprecated. Using frame {frame.frame}. "
                f"Instead, apply {self.__class__.__name__} to a frame."
            )
            frame = frame.frame

        if not isinstance(frequency, ParameterExpression):
            frequency = float(frequency)
        super().__init__(operands=(frequency, frame), name=name)

    @property
    def frequency(self) -> Union[float, ParameterExpression]:
        """New frequency."""
        return self.operands[0]

    @property
    def channel(self) -> PulseChannel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` or frame that this instruction is
        scheduled on.
        """
        if self.frame.prefix == "d":
            return DriveChannel(self.frame.index)

        if self.frame.prefix == "m":
            return MeasureChannel(self.frame.index)

        if self.frame.prefix == "u":
            return ControlChannel(self.frame.index)

    @property
    def frame(self) -> Frame:
        """Return the frame on which this instruction applies."""
        return self.operands[1]

    @property
    def channels(self) -> Tuple[PulseChannel]:
        """Returns the channels that this schedule uses."""
        if self.channel is not None:
            return (self.channel,)

        return tuple()

    @property
    def frames(self) -> Tuple[Frame]:
        """Return the frames this instructions acts on."""
        return (self.frame,)

    @property
    def duration(self) -> int:
        """Duration of this instruction."""
        return 0

    def is_parameterized(self) -> bool:
        """Return True iff the instruction is parameterized."""
        return isinstance(self.frequency, ParameterExpression) or self.frame.is_parameterized()


class ShiftFrequency(Instruction):
    """Shift the channel frequency away from the current frequency."""

    def __init__(
        self,
        frequency: Union[float, ParameterExpression],
        frame: Union[PulseChannel, Frame],
        name: Optional[str] = None,
    ):
        """Creates a new shift frequency instruction.

        Args:
            frequency: Frequency shift of the channel in Hz.
            frame: The frame or channel this instruction operates on.
            name: Name of this set channel frequency instruction.
        """
        if isinstance(frame, PulseChannel):
            warnings.warn(
                f"Applying {self.__class__.__name__} to channel {frame}. This "
                f"functionality will be deprecated. Using frame {frame.frame}. "
                f"Instead, apply {self.__class__.__name__} to a frame."
            )
            frame = frame.frame

        if not isinstance(frequency, ParameterExpression):
            frequency = float(frequency)
        super().__init__(operands=(frequency, frame), name=name)

    @property
    def frequency(self) -> Union[float, ParameterExpression]:
        """Frequency shift from the set frequency."""
        return self.operands[0]

    @property
    def channel(self) -> PulseChannel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` or frame that this instruction is
        scheduled on.
        """
        if self.frame.prefix == "d":
            return DriveChannel(self.frame.index)

        if self.frame.prefix == "m":
            return MeasureChannel(self.frame.index)

        if self.frame.prefix == "u":
            return ControlChannel(self.frame.index)

    @property
    def frame(self) -> Frame:
        """Return the frame on which this instruction applies."""
        return self.operands[1]

    @property
    def channels(self) -> Tuple[PulseChannel]:
        """Returns the channels that this schedule uses."""
        if self.channel is not None:
            return (self.channel,)

        return tuple()

    @property
    def frames(self) -> Tuple[Frame]:
        """Return the frames this instructions acts on."""
        return (self.frame,)

    @property
    def duration(self) -> int:
        """Duration of this instruction."""
        return 0

    def is_parameterized(self) -> bool:
        """Return True iff the instruction is parameterized."""
        return isinstance(self.frequency, ParameterExpression) or self.frame.is_parameterized()
