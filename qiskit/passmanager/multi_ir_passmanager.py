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

"""A staged passmanager supporting multiple IRs in the execution flow."""

from typing import Any, Generic
from collections.abc import Iterable

from .compilation_status import WorkflowStatus, PropertySet
from .base_tasks import Task, IR, IR_OUT, PassManagerState, Callback
from .passmanager import BasePassManager
from .flow_controllers import FlowControllerLinear


class MultiStagePassManager(Generic[IR, IR_OUT]):
    """A multi-staged pass manager supporting multiple IRs.

    Stages are BasePassManager or Task, or list thereof. BasePassManager tasks are executed only,
    the `_passmanager_frontend` and `_passmanager_backend` conversions are ignored.

    The current execution model linearizes the pass into a :class:`.FlowControllerLinear`
    to execute the tasks. This underlying model is subject to change and it is unsafe to
    build on this assumption. The public interfaces of this class, however, are stable.
    """

    def __init__(
        self,
        **stages: BasePassManager[Any] | Task[Any, Any] | Iterable[Task[Any, Any]],
    ):
        """
        Args:
            **stages: The stages as pass managers. These will be executed in the provided order
                and must have compatible IRs.
        """
        # we store the stages as attributes, plus keep track of the stages
        stage_names = []
        for name, stage in stages.items():
            stage_names.append(name)
            self.__setattr__(name, stage)

        # the stages are immutable, once set
        self._stages = tuple(stage_names)

    @property
    def stages(self) -> tuple[str]:
        """The stage names. These are immutable.

        The stages themselves can be modified by accessing the attribute with the same name
        as the stage.
        """
        return self._stages

    def to_flow_controller(self) -> FlowControllerLinear[IR, IR_OUT]:
        """Convert this multi-staged pass manager to a linear flow controller.

        This conversion normalizes this pass manager into a ``Task[IR, IR_OUT]`` and allows
        it to be nested inside a :class:`.MultiStagePassManager` itself or other execution flows.
        """
        tasks = []
        for name in self.stages:
            stage = getattr(self, name)
            if isinstance(stage, BasePassManager):
                tasks.extend(stage.to_flow_controller().tasks)
            elif isinstance(stage, Task):
                tasks.append(stage)
            else:  # case: Iterable[Task]
                tasks.extend(stage)

        return FlowControllerLinear(tasks)

    def run(
        self,
        in_programs: IR | Iterable[IR],
        callback: Callback[Any] | None = None,
        *,
        property_set: PropertySet | None = None,
    ) -> IR_OUT | Iterable[IR_OUT]:
        """Run the pass manager on a set of input programs.

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

        flow_controller = self.to_flow_controller()
        out_programs = _run_flow_controller(flow_controller, in_programs, state, callback)

        return out_programs


def _run_flow_controller(controller, programs, state, callback):
    """A helper to run FlowControllerLinear on program or an iterable of programs."""
    if isinstance(programs, Iterable):
        return list(map(_run_flow_controller, programs))

    return controller.execute(programs, state, callback)[0]
