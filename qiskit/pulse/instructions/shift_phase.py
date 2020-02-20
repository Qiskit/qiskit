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

"""The shift phase instruction updates the modulation phase of pulses played on a channel."""

import warnings

from typing import List, Optional, Union

from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.exceptions import PulseError
from .instruction import Instruction


class ShiftPhase(Instruction):
    """The shift phase instruction updates the modulation phase of proceeding pulses played on the
    same :py:class:`~qiskit.pulse.channels.Channel`. It is a relative increase in phase determined
    by the ``phase`` operand.

    The qubit phase is tracked in software, enabling instantaneous, error-free Z-rotations by using
    a ShiftPhase to update the frame tracking the qubit state.
    """

    def __init__(self, phase: float,
                 channel: Optional[PulseChannel] = None,
                 name: Optional[str] = None):
        """Instantiate a shift phase instruction, increasing the output signal phase on ``channel``
        by ``phase`` [radians].

        Args:
            phase: The rotation angle in radians.
            channel: The channel this instruction operates on.
            name: Display name for this instruction.
        """
        if channel is None:
            warnings.warn("Usage of ShiftPhase without specifying a channel is deprecated. For "
                          "example, ShiftPhase(3.14)(DriveChannel(0)) should be replaced by "
                          "ShiftPhase(3.14, DriveChannel(0)).", DeprecationWarning)
        self._phase = phase
        self._channel = channel
        super().__init__(0, channel, name=name or 'fc{:x}'.format(hash((phase, channel)))[:10])

    @property
    def phase(self) -> float:
        """Return the rotation angle enacted by this instruction in radians."""
        return self._phase

    @property
    def operands(self) -> List[Union[float, PulseChannel]]:
        """Return a list of instruction operands."""
        return [self.phase, self.channel]

    @property
    def channel(self) -> PulseChannel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on.
        """
        return self._channel

    def __call__(self, channel: PulseChannel) -> 'ShiftPhase':
        """Return a new ShiftPhase instruction supporting the deprecated syntax of FrameChange.

        Args:
            channel: The channel this instruction operates on.

        Raises:
            PulseError: If channel was already added.

        Returns:
            New ShiftPhase with both phase (from ``self`) and the ``channel`` provided.
        """
        if self._channel is not None:
            raise PulseError("The channel has already been assigned as {}.".format(self.channel))
        warnings.warn("Usage of ShiftPhase without specifying a channel is deprecated. For "
                      "example, ShiftPhase(3.14)(DriveChannel(0)) should be replaced by "
                      "ShiftPhase(3.14, DriveChannel(0)).", DeprecationWarning)
        return ShiftPhase(self.phase, channel)

    def __repr__(self):
        return "{}({}, {}{})".format(self.__class__.__name__,
                                     self.phase,
                                     self.channel,
                                     ", name={}".format(self.name) if self.name else "")
