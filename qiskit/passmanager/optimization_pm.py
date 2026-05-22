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

"""A pass manager designed for optimization within a single IR."""

from collections.abc import Iterable
from typing import Generic, TypeVar

from .base_tasks import Task, IR, Callback, PassManagerState, PropertySet
from .exceptions import PassManagerError
from .flow_controllers import FlowControllerLinear


class OptimizationPassManager(Task[IR, IR], Generic[IR]):
    """Execute a series of tasks, remaining in a single IR."""

    def __init__(self, tasks: Iterable[Task[IR, IR]] | None) -> None:
        if tasks is None:
            tasks = []

        self._tasks = list(tasks)  # normalize to a list

    def execute(
        self, passmanager_ir: IR, state: PassManagerState, callback: Callback[IR] | None = None
    ) -> IR:
        for task in self._tasks:
            passmanager_ir = task.execute(passmanager_ir, state, callback)
        return super().execute(passmanager_ir, state, callback)

    def run(self, in_programs: IR | Iterable[IR]) -> IR | Iterable[IR]:
        state = PassManagerState()

        if isinstance(in_programs, IR):
            return self.execute(in_programs, state, None)

        return list(map(self.run, in_programs))

    @property
    def tasks(self) -> list[Task[IR, IR]]:
        return self._tasks

    def append(
        self,
        tasks: Task[IR, IR] | Iterable[Task[IR, IR]],
    ) -> None:
        """Append tasks to the schedule of passes.

        Args:
            tasks: A set of pass manager tasks to be added to schedule.

        Raises:
            TypeError: When any element of tasks is not a subclass of passmanager Task.
        """
        if isinstance(tasks, Task):
            tasks = [tasks]
        if any(not isinstance(t, Task) for t in tasks):
            raise TypeError("Added tasks are not all valid pass manager task types.")

        self._tasks.append(tasks)

    def replace(
        self,
        index: int,
        tasks: Task[IR, IR] | Iterable[Task[IR, IR]],
    ) -> None:
        """Replace a particular pass in the scheduler.

        Args:
            index: Task index to replace, based on the position in :meth:`tasks`
            tasks: A set of pass manager tasks to be added to schedule.

        Raises:
            TypeError: When any element of tasks is not a subclass of passmanager Task.
            PassManagerError: If the index is not found.
        """
        try:
            self._tasks[index] = tasks
        except IndexError as ex:
            raise PassManagerError(f"Index to replace {index} does not exist") from ex

    def remove(self, index: int) -> None:
        """Removes a particular pass in the scheduler.

        Args:
            index: Pass index to remove, based on the position in :meth:`passes`.

        Raises:
            PassManagerError: If the index is not found.
        """
        try:
            del self._tasks[index]
        except IndexError as ex:
            raise PassManagerError(f"Index to remove {index} does not exist") from ex
