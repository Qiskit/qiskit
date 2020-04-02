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

"""An instruction for blocking time on a channel; useful for scheduling alignment."""
import warnings

from typing import Optional

from ..channels import Channel
from ..exceptions import PulseError
from .instruction import Instruction


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

    def __init__(self, duration: int,
                 channel: Optional[Channel] = None,
                 name: Optional[str] = None):
        """Create a new delay instruction.

        No other instruction may be scheduled within a ``Delay``.

        Args:
            duration: Length of time of the delay in terms of dt.
            channel: The channel that will have the delay.
            name: Name of the delay for display purposes.
        """
        if channel is None:
            warnings.warn("Usage of Delay without specifying a channel is deprecated. For "
                          "example, Delay(5)(DriveChannel(0)) should be replaced by "
                          "Delay(5, DriveChannel(0)).", DeprecationWarning)
        self._channel = channel
        super().__init__((duration, channel), duration, (channel,), name=name)

    @property
    def channel(self) -> Channel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on.
        """
        return self._channel

    def __call__(self, channel: Channel) -> 'Delay':
        """Return new ``Delay`` that is fully instantiated with both ``duration`` and a ``channel``.

        Args:
            channel: The channel that will have the delay.

        Return:
            Complete and ready to schedule ``Delay``.

        Raises:
            PulseError: If ``channel`` has already been set.
        """
        warnings.warn("Calling Delay with a channel is deprecated. Instantiate the delay with "
                      "a channel instead.", DeprecationWarning)
        if self._channel is not None:
            raise PulseError("The channel has already been assigned as {}.".format(self.channel))
        return Delay(self.duration, channel)
