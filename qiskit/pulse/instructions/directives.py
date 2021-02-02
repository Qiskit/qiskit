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
from typing import Optional, Union, Dict

from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
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
    used in the pulse builder context. Note that this is not an user-facing instruction,
    but implicitly applied for improvement of the program representation.

    This instruction clearly indicates the attached schedule is a subroutine
    that is defined outside the current scope. This instruction benefits the compiler
    to reuse the defined subroutines rather than redefining it multiple times.
    """

    # note that we cannot type hint for this due to cyclic import
    def __init__(self, subprogram):
        """Create a new call directive with subprogram.

        Note that the subprogram will not be further optimized or scheduled because
        this is predefined schedule outside the scope of current program.
        Though the structure of subprogram is preserved, we can assign arbitrary parameter
        because we can manage parameter values in individual subroutine with
        unique Parameter object.

        Args:
            subprogram (Schedule): A subprogram to wrap with call instruction.
        """
        super().__init__((subprogram, ), None,
                         channels=tuple(subprogram.channels),
                         name=subprogram.name)

        if subprogram.is_parameterized():
            for value in subprogram.parameters:
                if isinstance(value, ParameterExpression):
                    for param in value.parameters:
                        # Table maps parameter to operand index, 0 for ``subprogram``
                        self._parameter_table[param].append(0)

    @property
    def duration(self) -> Union[int, ParameterExpression]:
        """Duration of this instruction."""
        return self.operands[0].duration

    @property
    def subprogram(self):
        """Return attached subprogram.

        Returns:
            (Schedule): Attached schedule.
        """
        return self.operands[0]

    def assign_parameters(self,
                          value_dict: Dict[ParameterExpression, ParameterValueType]
                          ) -> 'Call':
        assigned_subprogram = self.subprogram.assign_parameters(value_dict)
        self._operands = (assigned_subprogram, )
        return self
