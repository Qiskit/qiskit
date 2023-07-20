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
from abc import ABC, abstractmethod
from collections.abc import Iterable, Callable, Iterator
from typing import Any, Protocol

from .exceptions import PassManagerError
from .propertyset import PropertySet, FencedPropertySet

logger = logging.getLogger(__name__)


class OptimizerTask(Protocol):
    """A definition of optimization task.

    The optimizer task takes a passmanager IR, and outputs new passmanager IR
    after program optimization. Optimization task can rely on the :class:`.PropertySet`
    to communicate intermediate state with other tasks.
    """

    @abstractmethod
    def execute(
        self,
        passmanager_ir: Any,
        property_set: PropertySet | None = None,
    ) -> Any:
        """Execute optimization task for input passmanager IR.

        Args:
            passmanager_ir: Passmanager IR to optimize.
            property_set: A local namespace associated with this optimization workflow.

        Returns:
            Optimized passmanager IR.
        """
        pass


class GenericPass(OptimizerTask, ABC):
    """Base class of a single optimization task.

    The optimization pass instance can read and write to the provided :class:`.PropertySet`.
    """

    def __init__(self):
        self.requires: Iterable[OptimizerTask] = []
        self.preserves: Iterable[GenericPass] = []
        self.property_set = PropertySet()

    def name(self) -> str:
        """Name of the pass."""
        return self.__class__.__name__

    def execute(
        self,
        passmanager_ir: Any,
        property_set: PropertySet | None = None,
    ) -> Any:
        self.property_set = property_set
        return self.run(passmanager_ir)

    @abstractmethod
    def run(
        self,
        passmanager_ir: Any,
    ) -> Any:
        """Run optimization task.

        Args:
            passmanager_ir: Passmanager IR to optimize.

        Returns:
            Optimized passmanager IR.
        """
        pass


class BaseFlowController(OptimizerTask, ABC):
    """Base class of flow controller.

    Flow controller is built with a list of optimizer tasks, and executes them with an input
    passmanager IR. Subclass must implement how the tasks are iterated over.
    Note that the flow controller can be nested into another flow controller,
    and flow controller itself doesn't provide any optimization subroutine.
    """

    def __init__(
        self,
        passes: list[OptimizerTask] | None = None,
        options: dict[str, Any] | None = None,
    ):
        """Create new flow controller.

        Args:
            passes: A list of optimization tasks.
            options: Option for this flow controller.
        """
        self._options = options or dict()
        self._property_set = PropertySet()
        self._callback = None

        self.pipeline: list[OptimizerTask] = passes

        # passes already run that have not been invalidated
        self.valid_passes = set()

    @property
    def property_set(self) -> PropertySet:
        """Property set of this flow controller."""
        return self._property_set

    @property
    def fenced_property_set(self) -> FencedPropertySet:
        """Readonly property set of this flow controller."""
        return FencedPropertySet(self._property_set)

    @property
    def callback(self) -> Callable:
        """A user provided function called per execution of single optimization task."""
        return self._callback

    @callback.setter
    def callback(self, new_callback: Callable):
        for task in self.pipeline:
            if isinstance(task, BaseFlowController):
                task.callback = new_callback
        self._callback = new_callback

    @abstractmethod
    def yield_pipeline(self) -> Iterator[OptimizerTask]:
        """Return iterator of optimization tasks."""
        pass

    def execute(
        self,
        passmanager_ir: Any,
        property_set: PropertySet | None = None,
    ):
        if property_set:
            self._property_set = property_set

        for task in self.yield_pipeline():
            if isinstance(task, GenericPass):
                for required in task.requires:
                    passmanager_ir = required.execute(passmanager_ir, self._property_set)
                if task not in self.valid_passes:
                    start_time = time.time()
                    try:
                        passmanager_ir = task.execute(passmanager_ir, self._property_set)
                    finally:
                        running_time = time.time() - start_time
                        log_msg = f"Pass: {task.name()} - {running_time * 1000:.5f} (ms)"
                        logger.info(log_msg)
                    self._finalize(
                        task=task,
                        passmanager_ir=passmanager_ir,
                        running_time=running_time,
                    )
                    return passmanager_ir

            if isinstance(task, BaseFlowController):
                return task.execute(passmanager_ir, self._property_set)

            raise PassManagerError(f"{task.__class__} is not a valid pass for flow controller.")

    def _finalize(
        self,
        task: GenericPass,
        passmanager_ir: Any,
        running_time: float,
    ):
        self.valid_passes.add(task)

        if self._callback is not None:
            self._callback(
                task=task,
                passmanager_ir=passmanager_ir,
                property_set=self._property_set,
                running_time=running_time,
            )


class ControllableController(BaseFlowController, ABC):
    """Base class of flow controller with controller callable.

    This is a special type of flow controller that iterates over optimization tasks
    based upon the evaluation of trigger condition. This condition is evaluated
    by a callback function to instantiate with, which consumes :class:`.PropertySet` and
    returns a boolean value.
    """

    def __init__(
        self,
        passes: list[OptimizerTask] | None = None,
        options: dict[str, Any] | None = None,
        controller_callback: Callable[[PropertySet], bool] = None,
    ):
        """Create new flow controller.

        Args:
            passes: A list of optimization tasks.
            options: Option for this flow controller.
            controller_callback: A callable to consume property set and provide a control.
        """
        super().__init__(passes, options)
        self.controller_callback = controller_callback
