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
from typing import Optional, Union

from qiskit.circuit import ParameterExpression
from qiskit.pulse import channels as chans, Schedule
from qiskit.pulse.instructions import Instruction


class Directive(Instruction, ABC):
    """A compiler directive.

    This is a hint to the pulse compiler and is not loaded into hardware.
    """

    @property
    def duration(self) -> int:
        """Duration of this instruction."""
        return 0


class RelativeBarrier(Directive):
    """Pulse ``RelativeBarrier`` directive."""

    def __init__(self,
                 *channels: chans.Channel,
                 name: Optional[str] = None):
        """Create a relative barrier directive.

        The barrier directive blocks instructions within the same schedule
        as the barrier on channels contained within this barrier from moving
        through the barrier in time.

        Args:
            channels: The channel that the barrier applies to.
            name: Name of the directive for display purposes.
        """
        super().__init__(tuple(channels), None, tuple(channels), name=name)

    def __eq__(self, other):
        """Verify two barriers are equivalent."""
        return (isinstance(other, type(self)) and
                set(self.channels) == set(other.channels))


class Call(Directive):
    """Pulse ``Call`` directive.

    This instruction wraps other instructions when ``pulse.call`` function is
    used in the pulse builder context.
    This instruction clearly indicates the attached program is a subroutine
    that is defined outside the current scope. This instruction benefits the compiler
    to reuse defined subroutines rather than redefining it multiple times.
    Having call instruction doesn't impact to the scheduling of instructions.
    """

    def __init__(self,
                 schedule: Schedule):
        """Create a new call directive.

        Args:
            schedule: Schedule to wrap with call instruction.
        """
        super().__init__((schedule, ), None, tuple(schedule.channels), name=name)

    @property
    def channel(self) -> Channel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on.
        """
        return self.operands[0].channel

    @property
    def duration(self) -> Union[int, ParameterExpression]:
        """Duration of this instruction."""
        return self.operands[0].duration
