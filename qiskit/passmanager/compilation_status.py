# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A property set dictionary that is shared among optimization passes."""

from dataclasses import dataclass, field
from enum import Enum
from qiskit._accelerate.passmanager import PassContext


class PropertySet(dict):
    """A default dictionary-like object."""

    def __missing__(self, key):
        return None


class RunState(Enum):
    """Allowed values for the result of a pass execution."""

    SUCCESS = 0
    FAIL = 1
    SKIP = 2


@dataclass
class WorkflowStatus:
    """Collection of compilation status of workflow, i.e. pass manager run.

    This data structure is initialized when the pass manager is run,
    and recursively handed over to underlying tasks.
    Each pass will update this status once after being executed, and the lifetime of the
    workflow status object is the time during which the pass manager is running.
    """

    count: int = 0
    """Current number of pass execution."""

    completed_passes: set = field(default_factory=set)
    """Passes already run that have not been invalidated."""

    previous_run: RunState = RunState.FAIL
    """Status of the latest pass run."""


@dataclass
class PassManagerState:
    """A portable container object that pass manager tasks communicate through generator.

    This object can contain all information about the running pass manager workflow,
    except for the IR object being optimized.
    The data structure consists of two elements; one for the status of the
    workflow itself, and another one for the additional information about the IR
    analyzed through pass executions. This container aims at just providing
    a robust interface for the :meth:`.Task.execute`, and no logic that modifies
    the container elements must be implemented.

    This object is mutable, and might be mutated by pass executions.
    """

    workflow_status: WorkflowStatus
    """Status of the current compilation workflow."""

    property_set: PropertySet | PassContext
    """Information about IR being optimized."""


def state_from_passcontext(context: PassContext) -> PassManagerState:
    """Create a PassManagerState from a PassContext for the new pass manager calling legacy passes.

    The workflow status is *not* tracked across passes and the new pass context is duck typed
    as property set.
    """
    empty_workflow_status = WorkflowStatus()
    return PassManagerState(empty_workflow_status, context)


def passcontext_from_state(state: PassManagerState) -> PassContext:
    if not isinstance(state.property_set, PassContext):
        raise TypeError("Cannot extract pass context from pass manager state.")
    return state.property_set
