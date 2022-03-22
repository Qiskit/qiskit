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

"""Directives are hints to the pulse compiler for how to process its input programs."""

from abc import ABC
from typing import Optional, Tuple

from qiskit.pulse import channels as chans
from qiskit.pulse.instructions import instruction


class Directive(instruction.Instruction, ABC):
    """A compiler directive.

    This is a hint to the pulse compiler and is not loaded into hardware.
    """

    @property
    def duration(self) -> int:
        """Duration of this instruction."""
        return 0


class RelativeBarrier(Directive):
    """Pulse ``RelativeBarrier`` directive."""

    def __init__(self, *channels: chans.Channel, name: Optional[str] = None):
        """Create a relative barrier directive.

        The barrier directive blocks instructions within the same schedule
        as the barrier on channels contained within this barrier from moving
        through the barrier in time.

        Args:
            channels: The channel that the barrier applies to.
            name: Name of the directive for display purposes.
        """
        super().__init__(operands=tuple(channels), name=name)

    @property
    def channels(self) -> Tuple[chans.Channel]:
        """Returns the channels that this schedule uses."""
        return self.operands

    def __eq__(self, other):
        """Verify two barriers are equivalent."""
        return isinstance(other, type(self)) and set(self.channels) == set(other.channels)
