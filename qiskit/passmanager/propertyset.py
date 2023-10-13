# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A property set dictionary that shared among optimization passes."""


from dataclasses import dataclass, field
from enum import IntEnum

from qiskit.utils.deprecation import deprecate_func
from .exceptions import PassManagerError


class PropertySet(dict):
    """A default dictionary-like object."""

    def __missing__(self, key):
        return None


class FencedPropertySet(PropertySet):
    """A readonly property set that cannot be written via __setitem__."""

    @deprecate_func(
        since="0.26.0",
        additional_msg=(
            "Internal use of FencedObject is already removed from pass manager. "
            "Implementation of a task subclass with protection for input object modification "
            "is now responsibility of the developer."
        ),
        pending=True,
    )
    def __init__(self, seq=None, **kwargs):
        super().__init__(seq, **kwargs)

    def __setitem__(self, key, value):
        raise PassManagerError("The fenced PropertySet has the property __setitem__ protected.")


class RunState(IntEnum):
    """Allowed values for the result of a pass execution."""

    SUCCESS = 0
    FAIL = 1
    SKIP = 2


@dataclass
class PassState:
    """Collection of compilation status of a single pass.

    This information is initialized in the pass manager instance at construction time,
    and recursively passed to flow controllers at running time.
    Pass will update the status once after being executed, and the status will alive
    during the execution of a single pass.
    """

    count: int = 0
    """Current number of pass execution."""

    completed_passes: set = field(default_factory=set)
    """Passes already run that have not been invalidated."""

    previous_run: RunState = RunState.FAIL
    """Status of the latest pass run."""
