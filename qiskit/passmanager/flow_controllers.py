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
from collections.abc import Callable, Iterable, Sequence, Generator
from typing import Type, Any

from qiskit.utils.deprecation import deprecate_func
from .base_tasks import (
    BaseController,
    Task,
)
from .propertyset import PropertySet
from .exceptions import PassManagerError

logger = logging.getLogger(__name__)


class FlowControllerMeta(type):
    """A magic to separate controller factory and flow control.

    FlowController provides a factory function to build a nested flow controller,
    but this class is also designed as a baseclass of all flow controllers.

    This structure causes unmanageable mix-up of factory function and flow control,
    yielding something awkward such that a user may instantiate a flow controller
    with FlowController.controller_factory to get its instance,
    and one can further build another controller instance from it.

    When a user subclasses the FlowController, there is no guarantee that
    the user always adds their controller to the base FlowController namespace.
    When a controller is added to the subclass, the base FlowController cannot
    look up the alias in its own namespace.

    This metaclass splits the responsibility of flow control from the FlowController
    and delegates it to the BaseFlowController, while allowing users to
    directly subclass the FlowController for "backward compatibility".

    Since the instance is considered as a subclass of the BaseFlowController,
    a functionality of controller factory is dropped from the FlowController instance.
    """

    def mro(cls) -> list[type]:
        return [cls, BaseController, Task, object]

    def __instancecheck__(cls, other):
        return isinstance(other, BaseController)


class FlowController(metaclass=FlowControllerMeta):
    """A flow controller with namespace to register controller subclasses.

    This allows syntactic suger of writing pipeline. For example,

    .. code-block:: python

        FlowController.add_flow_controller("my_condition", CustomController)

        controller = FlowController.controller_factory(
            [PassA(), PassB()],
            {"max_iteration": 1000},
            condition=lambda prop_set: prop_set["x"] == 0,
            do_while=lambda prop_set: prop_set["x"] < 100,
            my_condition=lambda prop_set: prop_set["y"] = "abc",
        )

    This creates a nested flow controller that runs when the value :code:`x` in the
    :class:`.PropertySet` is zero and repeats the pipeline until the value becomes 100.
    In each innermost loop, the custom iteration condition provided by
    the ``CustomController`` is also evaluated.
    """

    registered_controllers = {}
    hierarchy = []

    @classmethod
    def controller_factory(
        cls,
        passes: Task | list[Task],
        options: dict,
        **controllers,
    ):
        """Create new flow controller with normalization.

        Args:
            passes: A list of optimization tasks.
            options: Option for this flow controller.
            controllers: Dictionary of controller callables keyed on flow controller alias.

        Returns:
            An instance of normalized flow controller.
        """
        if None in controllers.values():
            raise PassManagerError("The controller needs a callable. Value cannot be None.")

        if isinstance(passes, BaseController):
            instance = passes
        else:
            instance = FlowControllerLinear(passes, options=options)

        if controllers:
            # Alias in higher hierarchy becomes outer controller.
            for alias in cls.hierarchy[::-1]:
                if alias not in controllers:
                    continue
                class_type = cls.registered_controllers[alias]
                init_kwargs = {
                    "options": options,
                    alias: controllers.pop(alias),
                }
                instance = class_type([instance], **init_kwargs)

        return instance

    @classmethod
    def add_flow_controller(
        cls,
        name: str,
        controller: Type[BaseController],
    ):
        """Adds a flow controller.

        Args:
            name: Alias of controller class in the namespace.
            controller: Flow controller class.
        """
        cls.registered_controllers[name] = controller
        if name not in cls.hierarchy:
            cls.hierarchy.append(name)

    @classmethod
    def remove_flow_controller(
        cls,
        name: str,
    ):
        """Removes a flow controller.

        Args:
            name: Alias of the controller to remove.

        Raises:
            KeyError: If the controller to remove was not registered.
        """
        if name not in cls.hierarchy:
            raise KeyError("Flow controller not found: %s" % name)
        del cls.registered_controllers[name]
        cls.hierarchy.remove(name)


class FlowControllerLinear(BaseController):
    """A standard flow controller that runs tasks one after the other."""

    def __init__(
        self,
        tasks: Task | Iterable[Task] = (),
        *,
        options: dict[str, Any] | None = None,
    ):
        super().__init__(options)

        if not isinstance(tasks, Sequence):
            tasks = [tasks]
        if any(not isinstance(t, Task) for t in tasks):
            raise PassManagerError("Added tasks are not all valid pass manager task types.")
        self.tasks: tuple[Task] = tuple(tasks)

    @property
    @deprecate_func(
        since="0.26.0",
        additional_msg="Use .tasks attribute instead.",
        is_property=True,
    )
    def passes(self) -> list[Task]:
        """Alias of pipeline for backward compatibility."""
        return list(self.tasks)

    @deprecate_func(
        since="0.26.0",
        additional_msg="All tasks must be provided at construction time of the controller object.",
    )
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

        tasks = list(self.tasks)
        for task in passes:
            if not isinstance(task, Task):
                raise PassManagerError(
                    f"New task {task} is not a valid pass manager pass or flow controller."
                )
            tasks.append(task)
        self.tasks = tuple(tasks)

    def iter_tasks(
        self,
        property_set: PropertySet,
    ) -> Generator[Task]:
        yield from self.tasks


class DoWhileController(BaseController):
    """A flow controller that repeatedly run the entire pipeline until the condition is not met."""

    def __init__(
        self,
        tasks: Task | Iterable[Task] = (),
        do_while: Callable[[PropertySet], bool] = None,
        *,
        options: dict[str, Any] | None = None,
    ):
        super().__init__(options)

        if not isinstance(tasks, Sequence):
            tasks = [tasks]
        if any(not isinstance(t, Task) for t in tasks):
            raise PassManagerError("Added tasks are not all valid pass manager task types.")
        self.tasks: tuple[Task] = tuple(tasks)
        self.do_while = do_while

    @property
    @deprecate_func(
        since="0.26.0",
        additional_msg="Use .tasks attribute instead.",
        is_property=True,
    )
    def passes(self) -> list[Task]:
        """Alias of pipeline for backward compatibility."""
        return list(self.tasks)

    @deprecate_func(
        since="0.26.0",
        additional_msg="All tasks must be provided at construction time of the controller object.",
    )
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

        tasks = list(self.tasks)
        for task in passes:
            if not isinstance(task, Task):
                raise PassManagerError(
                    f"New task {task} is not a valid pass manager pass or flow controller."
                )
            tasks.append(task)
        self.tasks = tuple(tasks)

    def iter_tasks(
        self,
        property_set: PropertySet,
    ) -> Generator[Task]:
        max_iteration = self._options.get("max_iteration", 1000)
        for _ in range(max_iteration):
            yield from self.tasks
            if not self.do_while(property_set):
                return
        raise PassManagerError("Maximum iteration reached. max_iteration=%i" % max_iteration)


class ConditionalController(BaseController):
    """A flow controller runs the pipeline once when the condition is met."""

    def __init__(
        self,
        tasks: Task | Iterable[Task] = (),
        condition: Callable[[PropertySet], bool] = None,
        *,
        options: dict[str, Any] | None = None,
    ):
        super().__init__(options)

        if not isinstance(tasks, Sequence):
            tasks = [tasks]
        if any(not isinstance(t, Task) for t in tasks):
            raise PassManagerError("Added tasks are not all valid pass manager task types.")
        self.tasks: tuple[Task] = tuple(tasks)
        self.condition = condition

    @property
    @deprecate_func(
        since="0.26.0",
        additional_msg="Use .tasks attribute instead.",
        is_property=True,
    )
    def passes(self) -> list[Task]:
        """Alias of pipeline for backward compatibility."""
        return list(self.tasks)

    @deprecate_func(
        since="0.26.0",
        additional_msg="All tasks must be provided at construction time of the controller object.",
    )
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

        tasks = list(self.tasks)
        for task in passes:
            if not isinstance(task, Task):
                raise PassManagerError(
                    f"New task {task} is not a valid pass manager pass or flow controller."
                )
            tasks.append(task)
        self.tasks = tuple(tasks)

    def iter_tasks(
        self,
        property_set: PropertySet,
    ) -> Generator[Task]:
        if self.condition(property_set):
            yield from self.tasks


FlowController.add_flow_controller("condition", ConditionalController)
FlowController.add_flow_controller("do_while", DoWhileController)
