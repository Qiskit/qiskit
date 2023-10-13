# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Baseclasses for the Qiskit passmanager optimization tasks."""
from __future__ import annotations

import logging
import time
from abc import abstractmethod, ABC
from collections.abc import Iterable, Callable, Generator
from typing import Any

from .exceptions import PassManagerError
from .propertyset import WorkflowStatus, PropertySet, RunState

logger = logging.getLogger(__name__)

# Type alias
PassManagerIR = Any


class Task(ABC):
    """An interface of the pass manager task.

    The task takes a Qiskit IR, and outputs new Qiskit IR after some operation on it.
    A task can rely on the :class:`.PropertySet` to communicate intermediate data among tasks.
    """

    @abstractmethod
    def execute(
        self,
        passmanager_ir: PassManagerIR,
        status: WorkflowStatus | None = None,
        property_set: PropertySet | None = None,
        callback: Callable = None,
    ) -> PassManagerIR:
        """Execute optimization task for input Qiskit IR.

        Args:
            passmanager_ir: Qiskit IR to optimize.
            status: Mutable status object of the whole pass manager workflow.
            property_set: A mutable data collection shared among all tasks.
            callback: A callback function which is caller per execution of optimization task.

        Returns:
            Optimized Qiskit IR.
        """
        pass


class GenericPass(Task, ABC):
    """Base class of a single pass manager task.

    A pass instance can read and write to the provided :class:`.PropertySet`,
    and may modify the input pass manager IR.
    """

    def __init__(self):
        self.requires: Iterable[Task] = []
        self.property_set = PropertySet()

    def name(self) -> str:
        """Name of the pass."""
        return self.__class__.__name__

    def execute(
        self,
        passmanager_ir: PassManagerIR,
        status: WorkflowStatus | None = None,
        property_set: PropertySet | None = None,
        callback: Callable = None,
    ) -> PassManagerIR:
        # Overriding this method is not safe.
        # Pass subclass must keep current implementation.
        # Especially, task execution may break when method signature is modified.

        if property_set is not None:
            self.property_set = property_set
        if status is None:
            status = WorkflowStatus()

        for required in self.requires:
            passmanager_ir = required.execute(
                passmanager_ir=passmanager_ir,
                status=status,
                property_set=self.property_set,
                callback=callback,
            )

        run_state = None
        ret = None
        start_time = time.time()
        try:
            if self not in status.completed_passes:
                ret = self.run(passmanager_ir)
                run_state = RunState.SUCCESS
            else:
                run_state = RunState.SKIP
        except Exception as ex:
            run_state = RunState.FAIL
            raise ex
        finally:
            ret = ret or passmanager_ir
            if run_state != RunState.SKIP:
                running_time = time.time() - start_time
                log_msg = f"Pass: {self.name()} - {running_time * 1000:.5f} (ms)"
                logger.info(log_msg)
                if callback is not None:
                    callback(
                        task=self,
                        passmanager_ir=ret,
                        property_set=self.property_set,
                        running_time=running_time,
                        count=status.count,
                    )
            self.update_status(status, run_state)
        return ret

    def update_status(
        self,
        status: WorkflowStatus,
        run_state: RunState,
    ):
        """Update workflow status.

        Args:
            status: Status object to mutably update.
            run_state: Completion status of current task.
        """
        status.previous_run = run_state
        if run_state == RunState.SUCCESS:
            status.count += 1
            status.completed_passes.add(self)

    @abstractmethod
    def run(
        self,
        passmanager_ir: PassManagerIR,
    ) -> PassManagerIR:
        """Run optimization task.

        Args:
            passmanager_ir: Qiskit IR to optimize.

        Returns:
            Optimized Qiskit IR.
        """
        pass


class BaseController(Task, ABC):
    """Base class of controller.

    A controller is built with a collection of pass manager tasks,
    and a subclass provides a custom logic to choose next task to run.
    Note a controller can be nested into another controller,
    and controller itself doesn't provide any subroutine to modify the input IR.
    """

    def __init__(
        self,
        options: dict[str, Any] | None = None,
    ):
        """Create new flow controller.

        Args:
            options: Option for this flow controller.
        """
        self._options = options or {}

    @abstractmethod
    def iter_tasks(
        self,
        property_set: PropertySet,
    ) -> Generator[Task]:
        """A custom logic to choose a next task to run.

        Controller subclass can consume the property set to build a proper task pipeline.
        This indicates the order of task execution is only determined at running time.

        Args:
            property_set: A mutable data collection shared among all tasks.

        Yields:
            Next task to run.
        """
        pass

    def execute(
        self,
        passmanager_ir: PassManagerIR,
        status: WorkflowStatus | None = None,
        property_set: PropertySet | None = None,
        callback: Callable = None,
    ) -> PassManagerIR:
        # Overriding this method is not safe.
        # Pass subclass must keep current implementation.
        # Especially, task execution may break when method signature is modified.

        if property_set is None:
            property_set = PropertySet()
        if status is None:
            status = WorkflowStatus()

        for next_task in self.iter_tasks(property_set=property_set):
            try:
                passmanager_ir = next_task.execute(
                    passmanager_ir=passmanager_ir,
                    status=status,
                    property_set=property_set,
                    callback=callback,
                )
            except TypeError as ex:
                raise PassManagerError(
                    f"{next_task.__class__} is not a valid pass for flow controller."
                ) from ex

        return passmanager_ir
