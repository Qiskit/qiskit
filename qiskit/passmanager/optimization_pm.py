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

from .compilation_status import WorkflowStatus, PropertySet
from .base_tasks import Task, IR, Callback, PassManagerState
from .exceptions import PassManagerError


class OptimizationPassManager(Task[IR, IR], Generic[IR]):
    """Execute a series of tasks, remaining in a single IR."""

    def __init__(self, tasks: Iterable[Task[IR, IR]] | None) -> None:
        """
        Args:
            tasks: The tasks to run in the pass manager. These must preserve the IR as in- and
                outputs.
        """
        if tasks is None:
            tasks = []

        self._tasks = list(tasks)  # normalize to a list

    def execute(
        self, passmanager_ir: IR, state: PassManagerState, callback: Callback[IR] | None = None
    ) -> tuple[IR, PassManagerState]:
        for task in self._tasks:
            passmanager_ir, state = task.execute(passmanager_ir, state, callback)
        return passmanager_ir, state

    def run(
        self,
        in_programs: IR | Iterable[IR],
        callback: Callback[IR] | None = None,
        *,
        property_set: PropertySet | None = None,
    ) -> IR | Iterable[IR]:
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

    @property
    def tasks(self) -> list[Task[IR, IR]]:
        """The tasks run in the pass manager."""
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

        self._tasks.extend(tasks)

    def replace(
        self,
        index: int,
        tasks: Task[IR, IR] | Iterable[Task[IR, IR]],
    ) -> None:
        """Replace a particular task in the pass manager.

        Args:
            index: Task index to replace, based on the position in :attr:`tasks`.
            tasks: A task (or set of tasks) to replace the existing task with.

        Raises:
            PassManagerError: If the index is not found.
        """
        try:
            self._tasks[index] = tasks
        except IndexError as ex:
            raise PassManagerError(f"Index to replace {index} does not exist") from ex

    def remove(self, index: int) -> None:
        """Removes a particular task in the pass manager.

        Args:
            index: Task index to remove, based on the position in :attr:`tasks`.

        Raises:
            PassManagerError: If the index is not found.
        """
        try:
            del self._tasks[index]
        except IndexError as ex:
            raise PassManagerError(f"Index to remove {index} does not exist") from ex
