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

from .base_tasks import Task, IR, IR_OUT, PassManagerState, Callback
from .exceptions import PassManagerError
from .optimization_pm import OptimizationPassManager
from .lowering_pm import LoweringPassManager


class MultiStagePassManager(Task[IR, IR_OUT], Generic[IR, IR_OUT]):
    """A multi-staged pass manager supporting multiple IRs."""

    def __init__(
        self, **stages: dict[str, OptimizationPassManager[Any] | LoweringPassManager[Any, Any]]
    ):
        # we store the stages as attributes, plus keep track of the stages
        stages = []
        for name, pm in stages:
            stages.append(name)
            self.__setattr__(name, pm)

        # the stages are immutable, once set
        self._stages = tuple(stages)

    @property
    def stages(self) -> tuple[str]:
        return self._stages

    def execute(
        self, passmanager_ir: IR, state: PassManagerState, callback: Callback[Any] | None = None
    ) -> IR_OUT:
        for stage in self.stages:
            pm: Task[Any, Any] = getattr(self, stage)

            # try dynamically verifying the IR types
            ir_out_type = None
            if (ir_types := try_determine_ir_types(pm)) is not None:
                ir_type, ir_out_type = ir_types
                if not isinstance(passmanager_ir, ir_type):
                    raise PassManagerError(
                        f"Incompatible IR input type {type(passmanager_ir)}. "
                        f"Stage {stage} expected {ir_type} as input type."
                    )

            passmanager_ir = pm.execute(passmanager_ir, state, callback)

            # verify the output type is correct
            if ir_out_type is not None:
                if not isinstance(passmanager_ir, ir_out_type):
                    raise PassManagerError(
                        f"Incompatible IR output type {type(passmanager_ir)}. "
                        f"Stage {stage} promised to produce {ir_out_type} as output type."
                    )

        return passmanager_ir

    def run(self, in_programs: IR | Iterable[IR]) -> IR_OUT | Iterable[IR_OUT]:
        state = PassManagerState()

        if isinstance(in_programs, Iterable):
            return list(map(self.run, in_programs))

        return self.execute(in_programs, state, None)


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
        if type(base) is Task:
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

    return types
