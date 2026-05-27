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

"""A pass manager designed for lowering from one IR to another."""

from collections.abc import Iterable
from typing import Generic

from .compilation_status import WorkflowStatus, PropertySet
from .base_tasks import Task, IR, IR_OUT, Callback, PassManagerState
from .optimization_pm import OptimizationPassManager


class LoweringPassManager(Task[IR, IR_OUT], Generic[IR, IR_OUT]):
    """Execute a lowering task, with optional pre- and post-lowering tasks."""

    def __init__(
        self,
        lowering: Task[IR, IR_OUT],
        *,
        pre: Iterable[Task[IR, IR]] | None = None,
        post: Iterable[Task[IR_OUT, IR_OUT]] | None = None,
    ) -> None:
        self._lowering = lowering
        self._pre = OptimizationPassManager(pre)
        self._post = OptimizationPassManager(post)

    @property
    def pre_lowering(self) -> OptimizationPassManager[IR]:
        """The pre-lowering tasks. These preserve the input IR.

        This is accessed and manipulated as :class:`.OptimizationPassManager`.
        """
        return self._pre

    @property
    def post_lowering(self) -> OptimizationPassManager[IR_OUT]:
        """The post-lowering tasks. These preserve output IR.

        This is accessed and manipulated as :class:`.OptimizationPassManager`.
        """
        return self._post

    @property
    def lowering(self) -> Task[IR, IR_OUT]:
        """The lowering task."""
        return self._lowering

    def execute(
        self,
        passmanager_ir: IR,
        state: PassManagerState,
        callback: Callback[IR | IR_OUT] | None = None,
    ) -> tuple[IR_OUT, PassManagerState]:
        passmanager_ir, state = self.pre_lowering.execute(passmanager_ir, state, callback)
        passmanager_ir, state = self.lowering.execute(passmanager_ir, state, callback)
        passmanager_ir, state = self.post_lowering.execute(passmanager_ir, state, callback)

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
