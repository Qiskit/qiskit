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

"""An instruction to transmit a given pulse on a ``PulseChannel`` (i.e., those which support
transmitted pulses, such as ``DriveChannel``).
"""
from typing import List, Optional, Union

from ..channels import PulseChannel
from .instruction import Instruction


class Play(Instruction):
    """An instruction specifying the exact time dynamics of the output signal envelope for a
    limited time on the channel that the instruction is operating on.

    This signal is modulated by a phase and frequency which are controlled by separate
    instructions. The pulse duration must be fixed, and is implicitly given in terms of the
    cycle time, dt, of the backend.
    """

    def __init__(self, pulse: 'Pulse', channel: PulseChannel, name: Optional[str] = None):
        """Create a new pulse instruction.

        Args:
            pulse: A pulse waveform description, such as
                   :py:class:`~qiskit.pulse.pulse_lib.SamplePulse`.
            channel: The channel to which the pulse is applied.
            name: Name of the delay for display purposes.
        """
        self._pulse = pulse
        self._channel = channel
        super().__init__(pulse.duration, channel, name=name if name is not None else pulse.name)

    @property
    def operands(self) -> List[Union['Pulse', PulseChannel]]:
        """Return a list of instruction operands."""
        return [self.pulse, self.channel]

    @property
    def pulse(self) -> 'Pulse':
        """A description of the samples that will be played; for instance, exact sample data or
        a known function like Gaussian with parameters.
        """
        return self._pulse

    @property
    def channel(self) -> PulseChannel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on.
        """
        return self._channel

    def __repr__(self):
        return "{}({}, {}, name={})".format(self.__class__.__name__, self.pulse, self.channel,
                                            self.name)
