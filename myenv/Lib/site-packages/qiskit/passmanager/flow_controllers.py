# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019, 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Built-in pass flow controllers."""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Generator
from typing import Any

from .base_tasks import BaseController, Task
from .compilation_status import PassManagerState, PropertySet
from .exceptions import PassManagerError

logger = logging.getLogger(__name__)


class FlowControllerLinear(BaseController):
    """A standard flow controller that runs tasks one after the other."""

    def __init__(
        self,
        tasks: Task | Iterable[Task] = (),
        *,
        options: dict[str, Any] | None = None,
    ):
        super().__init__(options)

        if not isinstance(tasks, Iterable):
            tasks = [tasks]
        self.tasks: tuple[Task] = tuple(tasks)

    @property
    def passes(self) -> list[Task]:
        """Alias of tasks for backward compatibility."""
        return list(self.tasks)

    def iter_tasks(self, state: PassManagerState) -> Generator[Task, PassManagerState, None]:
        for task in self.tasks:
            state = yield task


class DoWhileController(BaseController):
    """Run the given tasks in a loop until the ``do_while`` condition on the property set becomes
    ``False``.

    The given tasks will always run at least once, and on iteration of the loop, all the
    tasks will be run (with the exception of a failure state being set)."""

    def __init__(
        self,
        tasks: Task | Iterable[Task] = (),
        do_while: Callable[[PropertySet], bool] = None,
        *,
        options: dict[str, Any] | None = None,
    ):
        super().__init__(options)

        if not isinstance(tasks, Iterable):
            tasks = [tasks]
        self.tasks: tuple[Task] = tuple(tasks)
        self.do_while = do_while

    @property
    def passes(self) -> list[Task]:
        """Alias of tasks for backward compatibility."""
        return list(self.tasks)

    def iter_tasks(self, state: PassManagerState) -> Generator[Task, PassManagerState, None]:
        max_iteration = self._options.get("max_iteration", 1000)
        for _ in range(max_iteration):
            for task in self.tasks:
                state = yield task
            if not self.do_while(state.property_set):
                return
            # Remove stored tasks from the completed task collection for next loop
            state.workflow_status.completed_passes.difference_update(self.tasks)
        raise PassManagerError(f"Maximum iteration reached. max_iteration={max_iteration}")


class ConditionalController(BaseController):
    """A flow controller runs the pipeline once if the condition is true, or does nothing if the
    condition is false."""

    def __init__(
        self,
        tasks: Task | Iterable[Task] = (),
        condition: Callable[[PropertySet], bool] = None,
        *,
        options: dict[str, Any] | None = None,
    ):
        super().__init__(options)

        if not isinstance(tasks, Iterable):
            tasks = [tasks]
        self.tasks: tuple[Task] = tuple(tasks)
        self.condition = condition

    @property
    def passes(self) -> list[Task]:
        """Alias of tasks for backward compatibility."""
        return list(self.tasks)

    def iter_tasks(self, state: PassManagerState) -> Generator[Task, PassManagerState, None]:
        if self.condition(state.property_set):
            for task in self.tasks:
                state = yield task
