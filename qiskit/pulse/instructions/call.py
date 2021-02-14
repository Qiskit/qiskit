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

"""Call instruction that represents calling a schedule as a subroutine."""

from typing import Optional, Union, Dict, Tuple, Any, Set

from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.instructions import instruction

# TODO This instruction should support ScheduleBlock when it's ready.


class Call(instruction.Instruction):
    """Pulse ``Call`` instruction.

    The ``Call`` instruction represents the calling of a referenced subroutine (schedule).
    It enables code reuse both within the pulse representation and hardware (if supported).
    Note that this instruction is not exposed to end users.
    """
    # Prefix to use for auto naming.
    prefix = 'call'

    def __init__(self, subroutine, name: Optional[str] = None):
        """Define new subroutine.

        .. note:: Inline subroutine is mutable. This requires special care for modification.

        Args:
            subroutine (Schedule): A program subroutine to be referred to.
            name: Unique ID of this subroutine. If not provided, this is generated based on
                the hash of instructions of the subroutine.
        """
        if name is None:
            name = f"{self.prefix}_{subroutine.name}"

        super().__init__((subroutine,), None,
                         channels=tuple(subroutine.channels),
                         name=name)

    @property
    def duration(self) -> Union[int, ParameterExpression]:
        """Duration of this instruction."""
        return self.subroutine.duration

    # pylint: disable=missing-return-type-doc
    @property
    def subroutine(self):
        """Return attached subroutine.

        Returns:
            schedule (Schedule): Attached schedule.
        """
        return self.operands[0]

    def _initialize_parameter_table(self,
                                    operands: Tuple[Any]):
        """A helper method to initialize parameter table.

        The behavior of the parameter table of the ``Call`` instruction is slightly different from
        other instructions. The actual parameter mapper object is defined only in the
        subroutine, thus the call instruction doesn't have operand of ``ParameterExpression`` type.
        The parameter table is defined as a mapping of parameter objects to assigned values,
        whereas the standard instruction stores the mapping to the operand tuple index.

        Note that this instruction doesn't immediately bind parameter values when the
        :meth:`assign_parameters` method is called with the parameter dictionary.
        Instead, this instruction separately keeps the parameter values from the subroutine.
        This logic enables the compiler to reuse the subroutine with different parameters.

        Args:
            operands: List of operands associated with this instruction.
        """
        if operands[0].is_parameterized():
            for value in operands[0].parameters:
                self._parameter_table[value] = None

    def assign_parameters(self,
                          value_dict: Dict[ParameterExpression, ParameterValueType]
                          ) -> 'Call':
        """Store parameters which will be later assigned to the subroutine.

        Parameter values are not immediately assigned. The subroutine with parameters
        assigned according to the populated parameter table will be generated only when
        :func:`~qiskit.pulse.transforms.inline_subroutines` function is applied to this
        instruction. Note that parameter assignment logic creates a copy of subroutine
        to avoid the mutation problem. This function is usually applied by the Qiskit
        compiler when the program is submitted to the backend.

        Args:
            value_dict: A mapping from Parameters to either numeric values or another
                Parameter expression.

        Returns:
            Self with updated parameters.
        """
        for param_obj, assigned_value in value_dict.items():
            if param_obj in self._parameter_table:
                self._parameter_table[param_obj] = assigned_value

        return self

    def is_parameterized(self) -> bool:
        """Return True iff the instruction is parameterized."""
        return any([value is None for value in self.arguments.values()])

    @property
    def parameters(self) -> Set:
        """Parameters which determine the instruction behavior."""
        return set(key for key, val in self._parameter_table.items() if val is None)

    @property
    def arguments(self) -> Dict[ParameterExpression, ParameterValueType]:
        """Parameters dictionary which determine the subroutine behavior."""
        return self._parameter_table
