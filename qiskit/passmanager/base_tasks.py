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

from .exceptions import PassManagerError
from .propertyset import PassState, PropertySet, FencedPropertySet, RunState

logger = logging.getLogger(__name__)


class Task(ABC):
    """An interface of the pass manager task.

    The task takes a Qiskit IR, and outputs new Qiskit IR after some operation on it.
    A task can rely on the :class:`.PassState` to communicate intermediate state among tasks.
    """

    _state: PassState | None

    @property
    def state(self) -> PassState:
        """A local state information associated with this optimization workflow."""
        if self._state is None:
            # Allocate state information as needed basis.
            # This slightly reduces memory footprint for instantiation.
            self._state = PassState()
        return self._state

    @property
    def property_set(self) -> PropertySet:
        """Property set of this flow controller."""
        return self.state.property_set

    @property_set.setter
    def property_set(self, new_property_set: PropertySet):
        self.state.property_set = new_property_set

    @abstractmethod
    def execute(
        self,
        passmanager_ir: Any,
        state: PassState | None = None,
        callback: Callable = None,
    ) -> Any:
        """Execute optimization task for input Qiskit IR.

        Args:
            passmanager_ir: Qiskit IR to optimize.
            state: A local state information associated with this optimization workflow.
            callback: A callback function which is caller per execution of optimization task.

        Returns:
            Optimized Qiskit IR.
        """
        pass


class GenericPass(Task, ABC):
    """Base class of a single optimization task.

    The optimization pass instance can read and write to the provided :class:`.PropertySet`.
    """

    def __init__(self):
        self.requires: Iterable[Task] = []
        self._state = None

    def name(self) -> str:
        """Name of the pass."""
        return self.__class__.__name__

    def execute(
        self,
        passmanager_ir: Any,
        state: PassState | None = None,
        callback: Callable = None,
    ) -> Any:
        if state is not None:
            self._state = state

        for required in self.requires:
            passmanager_ir = required.execute(
                passmanager_ir=passmanager_ir,
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
                    property_set=self.state.property_set,
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

    Flow controller is built with a list of optimizer tasks, and executes them with an input
    Qiskit IR. Subclass must implement how the tasks are iterated over.
    Note that the flow controller can be nested into another flow controller,
    and flow controller itself doesn't provide any optimization subroutine.
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
        self._state = None

        self.pipeline: list[Task] = []
        if passes:
            self.append(passes)

    @property
    def passes(self) -> list[Task]:
        """Alias of pipeline for backward compatibility."""
        return self.pipeline

    @property
    def fenced_property_set(self) -> FencedPropertySet:
        """Readonly property set of this flow controller."""
        return FencedPropertySet(self.state.property_set)

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
        state: PassState | None = None,
        callback: Callable = None,
    ):
        if state is not None:
            self._state = state

        for task in self:
            try:
                passmanager_ir = task.execute(
                    passmanager_ir=passmanager_ir,
                    state=self.state,
                    callback=callback,
                )
            except TypeError as ex:
                raise PassManagerError(
                    f"{task.__class__} is not a valid pass for flow controller."
                ) from ex

        return passmanager_ir
