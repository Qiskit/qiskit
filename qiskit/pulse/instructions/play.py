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
from typing import Optional

from ..channels import PulseChannel
from ..exceptions import PulseError
from ..library.pulse import Pulse
from .instruction import Instruction


class Play(Instruction):
    """This instruction is responsible for applying a pulse on a channel.

    The pulse specifies the exact time dynamics of the output signal envelope for a limited
    time. The output is modulated by a phase and frequency which are controlled by separate
    instructions. The pulse duration must be fixed, and is implicitly given in terms of the
    cycle time, dt, of the backend.
    """

    def __init__(self, pulse: Pulse,
                 channel: PulseChannel,
                 name: Optional[str] = None):
        """Create a new pulse instruction.

        Args:
            pulse: A pulse waveform description, such as
                   :py:class:`~qiskit.pulse.library.Waveform`.
            channel: The channel to which the pulse is applied.
            name: Name of the instruction for display purposes. Defaults to ``pulse.name``.

        Raises:
            PulseError: If pulse is not a Pulse type.
        """
        if not isinstance(pulse, Pulse):
            raise PulseError("The `pulse` argument to `Play` must be of type `library.Pulse`.")
        self._pulse = pulse
        self._channel = channel
        if name is None:
            name = pulse.name
        super().__init__((pulse, channel), pulse.duration, (channel,), name=name)

    @property
    def pulse(self) -> Pulse:
        """A description of the samples that will be played."""
        return self._pulse

    @property
    def channel(self) -> PulseChannel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on.
        """
        return self._channel
