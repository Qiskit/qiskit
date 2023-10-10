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
from collections.abc import Iterable, Callable, Iterator, Sequence
from typing import Any

from qiskit.utils.deprecation import deprecate_func
from .exceptions import PassManagerError
from .propertyset import PassState, PropertySet, FencedPropertySet, RunState

logger = logging.getLogger(__name__)


class Task(ABC):
    """An interface of the pass manager task.

    The task takes a Qiskit IR, and outputs new Qiskit IR after some operation on it.
    A task can rely on the :class:`.PropertySet` to communicate intermediate data among tasks.
    """

    property_set: PropertySet | None
    state: PassState | None

    @abstractmethod
    def execute(
        self,
        passmanager_ir: Any,
        property_set: PropertySet | None = None,
        state: PassState | None = None,
        callback: Callable = None,
    ) -> Any:
        """Execute optimization task for input Qiskit IR.

        Args:
            passmanager_ir: Qiskit IR to optimize.
            property_set: A mutable data collection shared among all tasks.
            state: A local state information associated with this optimization workflow.
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
        self.state = None
        self.property_set = None

    def name(self) -> str:
        """Name of the pass."""
        return self.__class__.__name__

    def execute(
        self,
        passmanager_ir: Any,
        property_set: PropertySet | None = None,
        state: PassState | None = None,
        callback: Callable = None,
    ) -> Any:
        # Overriding this method is not safe.
        # Pass subclass must keep current implementation.
        # Especially, task execution may break when method signature is modified.

        self.state = state or PassState()
        self.property_set = property_set if property_set is not None else PropertySet()

        for required in self.requires:
            passmanager_ir = required.execute(
                passmanager_ir=passmanager_ir,
                property_set=self.property_set,
                state=self.state,
                callback=callback,
            )
        if self in self.state.completed_passes:
            self.state.previous_run = RunState.SKIP
            return passmanager_ir

        start_time = time.time()
        ret = None
        try:
            ret = self.run(passmanager_ir)
            if ret is None:
                # Analysis pass may not return
                ret = passmanager_ir
        except Exception as ex:
            self.state.previous_run = RunState.FAIL
            raise ex
        finally:
            running_time = time.time() - start_time
            log_msg = f"Pass: {self.name()} - {running_time * 1000:.5f} (ms)"
            logger.info(log_msg)
            if callback is not None:
                callback(
                    task=self,
                    passmanager_ir=ret,
                    property_set=self.property_set,
                    running_time=running_time,
                    count=self.state.count,
                )
        self.state.previous_run = RunState.SUCCESS
        self.state.count += 1
        self.state.completed_passes.add(self)
        return ret

    @abstractmethod
    def run(
        self,
        passmanager_ir: Any,
    ) -> Any:
        """Run optimization task.

        Args:
            passmanager_ir: Qiskit IR to optimize.

        Returns:
            Optimized Qiskit IR.
        """
        pass


class BaseFlowController(Task, ABC):
    """Base class of flow controller.

    Flow controller is built with a list of pass manager tasks, and executes them with an input
    Qiskit IR. Subclass must implement how the tasks are iterated over.
    Note that the flow controller can be nested into another flow controller,
    and flow controller itself doesn't provide any subroutine to modify the IR.
    """

    def __init__(
        self,
        passes: Task | list[Task] | None = None,
        options: dict[str, Any] | None = None,
    ):
        """Create new flow controller.

        Args:
            passes: A list of optimization tasks or flow controller instance.
            options: Option for this flow controller.
        """
        self._options = options or {}
        self.state = None
        self.property_set = None

        self.pipeline: list[Task] = []
        if passes:
            self.append(passes)

    @property
    def passes(self) -> list[Task]:
        """Alias of pipeline for backward compatibility."""
        return self.pipeline

    @property
    @deprecate_func(
        since="0.26.0",
        pending=True,
        is_property=True,
    )
    def fenced_property_set(self) -> FencedPropertySet:
        """Readonly property set of this flow controller."""
        return FencedPropertySet(self.property_set)

    def __iter__(self) -> Iterator[Task]:
        raise NotImplementedError()

    def append(
        self,
        passes: Task | list[Task],
    ):
        """Add new task to pipeline.

        Args:
            passes: A new task or list of tasks to add.
        """
        if not isinstance(passes, Sequence):
            passes = [passes]
        for task in passes:
            if not isinstance(task, (GenericPass, BaseFlowController)):
                raise PassManagerError(
                    f"New task {task} is not a valid pass manager pass or flow controller."
                )
            self.pipeline.append(task)

    def execute(
        self,
        passmanager_ir: Any,
        property_set: PropertySet | None = None,
        state: PassState | None = None,
        callback: Callable = None,
    ):
        # Overriding this method is not safe.
        # Pass subclass must keep current implementation.
        # Especially, task execution may break when method signature is modified.

        self.state = state or PassState()
        self.property_set = property_set if property_set is not None else PropertySet()

        for task in self:
            try:
                passmanager_ir = task.execute(
                    passmanager_ir=passmanager_ir,
                    property_set=self.property_set,
                    state=self.state,
                    callback=callback,
                )
            except TypeError as ex:
                raise PassManagerError(
                    f"{task.__class__} is not a valid pass for flow controller."
                ) from ex

        return passmanager_ir
