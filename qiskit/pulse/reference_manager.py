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

"""Management of scehdule block reference."""

from typing import TYPE_CHECKING, Iterator, Sequence, Set, Tuple
from dataclasses import dataclass

from qiskit.pulse.utils import scoping_parameter
from qiskit.pulse.exceptions import PulseError

if TYPE_CHECKING:
    from qiskit.circuit.parameter import Parameter
    from .schedule import ScheduleBlock
    from .channels import Channel


@dataclass(frozen=True)
class SubroutineSpec:
    """Description of subroutine."""

    scope: str
    ref_key: str
    channels: Tuple["Channel", ...]
    schedule: "ScheduleBlock"

    @property
    def full_name(self) -> str:
        """Return full name of the subroutine."""
        return f"{self.scope}.{self.ref_key}"


class ReferenceManager:
    """Helper class to manage reference to subroutine."""

    def __init__(self):
        """Create new reference manager."""
        self._program_map = {}
        self._channel_map = {}

    def __contains__(self, item):
        if item in self._program_map:
            return True
        return False

    def __len__(self):
        return len(self._program_map)

    def add_key(
        self,
        ref_key: str,
        channels: Sequence["Channel"],
        scope: str,
    ):
        """Add new reference instruction.

        Args:
            ref_key: Reference key to the subroutine.
            channels: Channels associated with the subroutine.
            scope: Scope that subroutine exist.
        """
        scoped_key = f"{scope}.{ref_key}"
        if not scoped_key in self:
            self._program_map[scoped_key] = None
            self._channel_map[scoped_key] = channels

    def get_subroutine_with_key(
        self,
        ref_key: str,
        scope: str,
    ) -> "ScheduleBlock":
        """Get subroutine of corresponding reference instruction.

        Args:
            ref_key: Reference key to the subroutine.
            scope: Scope that subroutine exist.

        Returns:
            Assigned subtourine.

        Raises:
            PulseError: When subroutine of corresponding reference key is not found.
        """
        scoped_key = f"{scope}.{ref_key}"
        if scoped_key not in self:
            raise PulseError(f"Reference key {ref_key} does not exist in the scope {scope}.")
        return self._program_map[scoped_key]

    def assign_program_with_key(
        self,
        ref_key: str,
        scope: str,
        program: "ScheduleBlock",
    ):
        """Assign subroutine to reference instruction.

        Args:
            ref_key: Unique reference key of the subroutine.
            scope: Scope that subroutine exist.
            program: Schedule to assign.

        Raises:
            When subroutine of corresponding reference key is not found.
            PulseError: When channels don't match with the reference instruction.
            PulseError: When different subroutine is already assigned.
        """
        scoped_key = f"{scope}.{ref_key}"
        if scoped_key not in self:
            raise PulseError(f"Reference key {ref_key} does not exist in the scope {scope}.")

        # Check channel equality
        # Use channel-name based equality since channel index can be parametrized.
        # UUID based equality causes problem when channel indeces are separately created.
        sub_channels = set(c.name for c in program.channels)
        ref_channels = set(c.name for c in self._channel_map[scoped_key])
        if set(sub_channels) != set(ref_channels):
            raise PulseError(
                f"Channels of assigned program {sub_channels} are not identical to the "
                f"channels associated with the reference instruction {ref_channels}."
            )
        if self._program_map[scoped_key] is not None:
            if self._program_map[scoped_key] != program:
                raise PulseError(
                    f"Subroutine {ref_key} has been already assigned. "
                    "Newly assigned schedule has conflict with the previous assignment."
                )
        else:
            self._program_map[scoped_key] = program

    @property
    def parameters(self) -> Set["Parameter"]:
        """Returns parameters defined in the all subroutines."""
        sub_params = set()
        for subroutine in self._program_map.values():
            if subroutine is not None:
                sub_params = sub_params | subroutine.parameters
        return sub_params

    def scoped_parameters(self, current_scope: str) -> Set["Parameter"]:
        """Returns scoped parameters defined in the all subroutines.

        Args:
            current_scope: Name of current program scope.

        Returns:
            Set of parameter objects with updated name, while keeping UUID.
        """
        sub_params = set()
        for scoped_keys, subroutine in self._program_map.items():
            _, ref_key = scoped_keys.split(".")
            if subroutine is None:
                continue
            full_scoped_key = f"{current_scope}.{ref_key}"
            scoped_params = set()
            for param in subroutine.parameters:
                scoped_params.add(scoping_parameter(param, full_scoped_key))
            sub_params = sub_params | scoped_params
            if subroutine.is_referenced():
                nested_params = subroutine._references.scoped_parameters(full_scoped_key)
                sub_params = sub_params | nested_params
        return sub_params

    def get_reference_specs(self) -> Iterator[SubroutineSpec]:
        """Get specs of the all referened subroutines.

        Yields:
            Dataclass representing the spec of the subroutine.
        """
        for scoped_key in self._program_map:
            scope, ref_key = scoped_key.split(".")
            yield SubroutineSpec(
                scope=scope,
                ref_key=ref_key,
                channels=self._channel_map[scoped_key],
                schedule=self._program_map[scoped_key],
            )
