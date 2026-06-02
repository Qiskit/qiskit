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

from typing import Generic, Any
from collections.abc import Iterable

from .compilation_status import WorkflowStatus, PropertySet
from .base_tasks import Task, IR, IR_OUT, PassManagerState, Callback
from .preserving_pm import PreservingPassManager
from .lowering_pm import LoweringPassManager


class MultiStagePassManager(Task[IR, IR_OUT], Generic[IR, IR_OUT]):
    """A multi-staged pass manager supporting multiple IRs."""

    def __init__(
        self, **stages: dict[str, PreservingPassManager[Any] | LoweringPassManager[Any, Any]]
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
            passmanager_ir, stage = pm.execute(passmanager_ir, state, callback)

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
