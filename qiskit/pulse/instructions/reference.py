# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Call instruction that represents calling a schedule as a subroutine."""

from typing import Optional, Union, Tuple, Set

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.channels import Channel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.instructions import instruction


class Reference(instruction.Instruction):
    """Pulse compiler directive that refers to a subroutine.

    If a pulse program uses the same subset of instructions multiple times, then
    using the :class:`~.Reference` class may significantly reduce the memory footprint of
    the program. This instruction only stores the set of strings to identify the subroutine.

    The actual pulse program is stored in the :attr:`ScheduleBlock.references`
    that this reference instruction belongs to.

    You can later assign schedules with the :meth:`ScheduleBlock.assign_references` method.
    This allows you to build the main program without knowing the actual subroutine,
    that is supplied at a later time.
    """

    # Delimiter for representing nested scope.
    scope_delimiter = ":"

    def __init__(self, *ref_keys: str, name: Optional[str] = None):
        """Create new reference.

        Args:
            ref_keys: A set of string that represent this reference.
            name: Optional. A label of this reference.

        Raises:
            PulseError: When no key is provided.
            PulseError: When key is not a string.
            PulseError: When key contains scope delimiter.
        """
        if len(ref_keys) == 0:
            raise PulseError("At least one key must be provided.")

        if isinstance(ref_keys, str):
            ref_keys = (ref_keys,)

        # Run validation
        for key in ref_keys:
            if not isinstance(key, str):
                raise PulseError(f"Keys must be string. '{repr(key)}' is not a valid object.")
            if self.scope_delimiter in key:
                raise PulseError(
                    f"'{self.scope_delimiter}' is reserved. '{key}' is not valid key string."
                )

        super().__init__(operands=tuple(ref_keys), name=name)

    @property
    def ref_keys(self) -> Tuple[str, ...]:
        """Returns unique key of the subroutine."""
        return self.operands

    @property
    def duration(self) -> Union[int, ParameterExpression]:
        """Duration of this instruction."""
        return NotImplemented

    @property
    def channels(self) -> Tuple[Channel, ...]:
        """Returns the channels that this schedule uses."""
        return NotImplemented

    @property
    def parameters(self) -> Set:
        """Parameters which determine the instruction behavior."""
        return set()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ref_keys={self.ref_keys})"
