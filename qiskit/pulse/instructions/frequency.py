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

"""Frequency instructions module. These instructions allow the user to manipulate
the frequency of a channel.
"""
from typing import Optional

from ..channels import PulseChannel
from .instruction import Instruction


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

    def __init__(self, frequency: float,
                 channel: PulseChannel,
                 name: Optional[str] = None):
        """Creates a new set channel frequency instruction.

        Args:
            frequency: New frequency of the channel in Hz.
            channel: The channel this instruction operates on.
            name: Name of this set channel frequency instruction.
        """
        self._frequency = float(frequency)
        self._channel = channel
        super().__init__((frequency, channel), 0, (channel,), name=name)

    @property
    def frequency(self) -> float:
        """New frequency."""
        return self._frequency

    @property
    def channel(self) -> PulseChannel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on.
        """
        return self._channel


class ShiftFrequency(Instruction):
    """Shift the channel frequency away from the current frequency."""

    def __init__(self,
                 frequency: float,
                 channel: PulseChannel,
                 name: Optional[str] = None):
        """Creates a new shift frequency instruction.

        Args:
            frequency: Frequency shift of the channel in Hz.
            channel: The channel this instruction operates on.
            name: Name of this set channel frequency instruction.
        """
        self._frequency = float(frequency)
        self._channel = channel
        super().__init__((frequency, channel), 0, (channel,), name=name)

    @property
    def frequency(self) -> float:
        """Frequency shift from the set frequency."""
        return self._frequency

    @property
    def channel(self) -> PulseChannel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on.
        """
        return self._channel
