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
from typing import Generic, TypeVar

from .base_tasks import Task, IR, IR_OUT, Callback, PassManagerState, PropertySet
from .optimization_pm import OptimizationPassManager
from .exceptions import PassManagerError
from .flow_controllers import FlowControllerLinear


class LoweringPassManager(Task[IR, IR_OUT], Generic[IR, IR_OUT]):
    """Execute a series of tasks, remaining in a single IR."""

    def __init__(
        self,
        lowering: Task[IR, IR_OUT],
        *,
        pre: Iterable[Task[IR, IR]] | None,
        post: Iterable[Task[IR_OUT, IR_OUT]] | None,
    ) -> None:
        self._lowering = lowering
        self._pre = OptimizationPassManager(pre)
        self._post = OptimizationPassManager(post)

    def execute(
        self,
        passmanager_ir: IR,
        state: PassManagerState,
        callback: Callback[IR | IR_OUT] | None = None,
    ) -> IR_OUT:
        passmanager_ir = self._pre.execute(passmanager_ir, state, callback)
        passmanager_ir = self._lowering.execute(passmanager_ir, state, callback)
        passmanager_ir = self._post.execute(passmanager_ir, state, callback)

        return passmanager_ir

    def run(self, in_programs: IR | Iterable[IR]) -> IR_OUT | Iterable[IR_OUT]:
        state = PassManagerState()

        if isinstance(in_programs, IR):
            return self.execute(in_programs, state, None)

        return list(map(self.run, in_programs))
