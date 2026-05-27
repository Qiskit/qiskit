# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A multi-staged pass manager supporting multiple IRs."""

import typing
from typing import Generic, Any
from collections.abc import Iterable

from .compilation_status import WorkflowStatus, PropertySet
from .base_tasks import Task, IR, IR_OUT, PassManagerState, Callback
from .optimization_pm import OptimizationPassManager
from .lowering_pm import LoweringPassManager


class MultiStagePassManager(Task[IR, IR_OUT], Generic[IR, IR_OUT]):
    """A multi-staged pass manager supporting multiple IRs."""

    def __init__(
        self, **stages: dict[str, OptimizationPassManager[Any] | LoweringPassManager[Any, Any]]
    ):
        """
        Args:
            **stages: The stages as pass managers. These will be executed in the provided order
                and must have compatible IRs.
        """
        # we store the stages as attributes, plus keep track of the stages
        stage_names = []
        for name, pm in stages.items():
            stage_names.append(name)
            self.__setattr__(name, pm)

        # the stages are immutable, once set
        self._stages = tuple(stage_names)

    @property
    def stages(self) -> tuple[str]:
        """The stage names. These are immutable.

        The stages themselves can be modified by accessing the attribute with the same name
        as the stage.
        """
        return self._stages

    def execute(
        self, passmanager_ir: IR, state: PassManagerState, callback: Callback[Any] | None = None
    ) -> tuple[IR_OUT, PassManagerState]:
        for stage in self.stages:
            pm = getattr(self, stage)

            # try dynamically verifying the IR types
            # ir_in_type = None
            # ir_out_type = None
            # if (ir_types := try_determine_ir_types(pm)) is not None:
            #     ir_in_type, ir_out_type = ir_types
            # elif len(pm.tasks) > 0:
            #     if (first_task_types := try_determine_ir_types(pm.tasks[0])) is not None:
            #         ir_in_type, _ = first_task_types
            #     if (last_task_types := try_determine_ir_types(pm.tasks[-1])) is not None:
            #         _, ir_out_type = last_task_types

            # print("-- IN", ir_in_type)
            # print("-- OUT", ir_out_type)
            # if ir_in_type is not None:
            #     if not isinstance(passmanager_ir, ir_in_type):
            #         raise PassManagerError(
            #             f"Incompatible IR input type {type(passmanager_ir)}. "
            #             f"Stage {stage} expected {ir_in_type} as input type."
            #         )

            passmanager_ir, stage = pm.execute(passmanager_ir, state, callback)
            # print("ir:", passmanager_ir)

            # # verify the output type is correct
            # if ir_out_type is not None:
            #     if not isinstance(passmanager_ir, ir_out_type):
            #         raise PassManagerError(
            #             f"Incompatible IR output type {type(passmanager_ir)}. "
            #             f"Stage {stage} promised to produce {ir_out_type} as output type."
            #         )

        return passmanager_ir, state

    def run(
        self,
        in_programs: IR | Iterable[IR],
        callback: Callback[IR] | None = None,
        *,
        property_set: PropertySet | None = None,
    ) -> IR_OUT | Iterable[IR_OUT]:
        """Run the pass manager on a set of input programs.

        This is a convenience entry point to :meth:`run`, which allows to handle an iterable
        of input programs and creates a :class:`.PassManagerState` passed to the tasks.

        Args:
            in_programs: The programs to run the pass manager on.
            callback: A callback passed to each individual task.
            property_set: An optional property set to pass into the pass manager.

        Returns:
            The output programs.
        """
        if property_set is None:
            property_set = PropertySet()

        state = PassManagerState(property_set=property_set, workflow_status=WorkflowStatus())

        if isinstance(in_programs, Iterable):
            return list(map(self.run, in_programs))

        return self.execute(in_programs, state, callback)[0]


def try_determine_ir_types(task: Task) -> tuple[type, type] | None:
    """This function attempts to determine the IR types in a Task.

    This is based on typing.get_args and Type.__orig_bases__ or attribute.
    From Python 3.12 onwards we could use the API-documented ``get_original_bases``:
    https://docs.python.org/3/library/types.html#types.get_original_bases.
    """
    if (bases := getattr(task.__class__, "__orig_bases__", None)) is None:
        # could not determine the type
        return None

    task_base = None
    for base in bases:
        if typing.get_origin(base) is Task:
            task_base = base
            break

        # No ``Task`` is in the bases, meaning that either the input instance is invalid since
        # it doesn't derive from Task, or it doesn't derive from a Task with explicit Generics.
        # Either way, we cannot determine the IR types from this.
        return None

    types = typing.get_args(task_base)

    if len(types) != 2:
        # Failed to get the expected number of args. Task should have exactly 2 generic types.
        return None

    types = tuple(map(filter_type_var, types))
    if types[0] is None and types[1] is None:
        return None

    return types


def filter_type_var(the_type):
    if isinstance(the_type, typing.TypeVar):
        return None
    return the_type
