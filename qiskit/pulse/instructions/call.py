# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Call instruction that represents calling a schedule as a subroutine."""

from typing import Optional, Union, Dict, Tuple, Set

from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.channels import Channel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.instructions import instruction
from qiskit.utils.deprecation import deprecate_func


class Call(instruction.Instruction):
    """Pulse ``Call`` instruction.

    The ``Call`` instruction represents the calling of a referenced subroutine (schedule).
    It enables code reuse both within the pulse representation and hardware (if supported).
    """

    # Prefix to use for auto naming.
    prefix = "call"

    @deprecate_func(
        since="0.25.0",
        additional_msg="Instead, use the pulse builder function "
        "qiskit.pulse.builder.call(subroutine) within an active building context.",
    )
    def __init__(
        self,
        subroutine,
        value_dict: Optional[Dict[ParameterExpression, ParameterValueType]] = None,
        name: Optional[str] = None,
    ):
        """Define new subroutine.

        .. note:: Inline subroutine is mutable. This requires special care for modification.

        Args:
            subroutine (Union[Schedule, ScheduleBlock]): A program subroutine to be referred to.
            value_dict: Mapping of parameter object to assigned value.
            name: Unique ID of this subroutine. If not provided, this is generated based on
                the subroutine name.

        Raises:
            PulseError: If subroutine is not valid data format.
        """
        from qiskit.pulse.schedule import Schedule, ScheduleBlock

        if not isinstance(subroutine, (Schedule, ScheduleBlock)):
            raise PulseError(f"Subroutine type {subroutine.__class__.__name__} cannot be called.")

        value_dict = value_dict or {}

        # initialize parameter template
        if subroutine.is_parameterized():
            self._arguments = {par: value_dict.get(par, par) for par in subroutine.parameters}
            assigned_subroutine = subroutine.assign_parameters(
                value_dict=self.arguments, inplace=False
            )
        else:
            self._arguments = {}
            assigned_subroutine = subroutine

        # create cache data of parameter-assigned subroutine
        self._assigned_cache = (self._get_arg_hash(), assigned_subroutine)

        super().__init__(operands=(subroutine,), name=name or f"{self.prefix}_{subroutine.name}")

    @property
    def duration(self) -> Union[int, ParameterExpression]:
        """Duration of this instruction."""
        return self.subroutine.duration

    @property
    def channels(self) -> Tuple[Channel]:
        """Returns the channels that this schedule uses."""
        return self.assigned_subroutine().channels

    @property
    def subroutine(self):
        """Return attached subroutine.

        Returns:
            program (Union[Schedule, ScheduleBlock]): The program referenced by the call.
        """
        return self.operands[0]

    def assigned_subroutine(self):
        """Returns this subroutine with the parameters assigned.

        .. note:: This function may be often called internally for class equality check
            despite its overhead of parameter assignment.
            The subroutine with parameter assigned is cached based on ``.argument`` hash.
            Once this argument is updated, new assigned instance will be returned.
            Note that this update is not mutable operation.

        Returns:
            program (Union[Schedule, ScheduleBlock]): Attached program.
        """
        if self._get_arg_hash() != self._assigned_cache[0]:
            subroutine = self.subroutine.assign_parameters(value_dict=self.arguments, inplace=False)
            # update cache data
            self._assigned_cache = (self._get_arg_hash(), subroutine)
        else:
            subroutine = self._assigned_cache[1]

        return subroutine

    @property
    def parameters(self) -> Set:
        """Unassigned parameters which determine the instruction behavior."""
        params = set()
        for value in self._arguments.values():
            if isinstance(value, ParameterExpression):
                params |= value.parameters
        return params

    @property
    def arguments(self) -> Dict[ParameterExpression, ParameterValueType]:
        """Parameters dictionary to be assigned to subroutine."""
        return self._arguments

    @arguments.setter
    def arguments(self, new_arguments: Dict[ParameterExpression, ParameterValueType]):
        """Set new arguments.

        Args:
            new_arguments: Dictionary of new parameter value mapping to update.

        Raises:
            PulseError: When new arguments doesn't match with existing arguments.
        """
        # validation
        if new_arguments.keys() != self._arguments.keys():
            new_arg_names = ", ".join(map(repr, new_arguments.keys()))
            old_arg_names = ", ".join(map(repr, self.arguments.keys()))
            raise PulseError(
                "Key mismatch between new arguments and existing arguments. "
                f"{new_arg_names} != {old_arg_names}"
            )

        self._arguments = new_arguments

    def _get_arg_hash(self):
        """A helper function to generate hash of parameters."""
        return hash(tuple(self.arguments.items()))

    def __eq__(self, other: instruction.Instruction) -> bool:
        """Check if this instruction is equal to the `other` instruction.

        Instructions are equal if they share the same type, operands, and channels.
        """
        # type check
        if not isinstance(other, self.__class__):
            return False

        # compare subroutine. assign parameter values before comparison
        if self.assigned_subroutine() != other.assigned_subroutine():
            return False

        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.assigned_subroutine()}, name='{self.name}')"
