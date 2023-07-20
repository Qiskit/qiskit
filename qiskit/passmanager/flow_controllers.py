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
from collections.abc import Iterator
from typing import Type

from .base_optimization_tasks import (
    BaseFlowController,
    ControllableController,
    GenericPass,
    OptimizerTask,
)
from .exceptions import PassManagerError

logger = logging.getLogger(__name__)


class FlowController:
    """A flow controller namespace to instantiate flow controller with controller dictionary.

    This allows syntactic suger of writing pipeline. For example,

    .. code-block:: python

        controller = FlowController(
            [PassA(), PassB()],
            {"max_iteration": 1000},
            condition=lambda prop_set: prop_set["x"] == 0,
            do_while=lambda prop_set: prop_set["x"] < 100,
        )

    This creates a nested flow controller that runs when the value :code:`x` in the
    :class:`.PropertySet` is zero and repeats the pipeline until the value becomes 100.

    .. note::

        This class creates an instance of :class:`.BaseFlowController` rather than itself.
        Instance check must be done against :class:`.BaseFlowController` type.
        This class just accommodates a controller namespace to allow instantiation of
        the structured :class:`.ControllableController` with keyword arguments.
        User can also manually instantiate :class:`.ControllableController` instances
        with dependency.
    """

    registered_controllers = {}
    hierarchy = []

    def __new__(
        cls,
        passes: OptimizerTask | list[OptimizerTask],
        options: dict,
        **controllers,
    ) -> BaseFlowController:
        """Create new flow controller with normalization.

        Args:
            passes: A list of optimization tasks.
            options: Option for this flow controller.
            controllers: Dictionary of controller callables keyed on flow controller alias.

        Returns:
            An instance of normalized flow controller.
        """
        if isinstance(passes, BaseFlowController):
            return passes

        if isinstance(passes, GenericPass):
            passes = [passes]

        if None in controllers.values():
            raise PassManagerError("The controller needs a callable. Value cannot be None.")

        instance = FlowControllerLiner(passes, options)

        if controllers:
            # Alias in higher hierarchy becomes outer controller.
            for alias in cls.hierarchy[::-1]:
                class_type = cls.registered_controllers[alias]
                if alias not in controllers or not issubclass(class_type, ControllableController):
                    continue
                instance = class_type(
                    passes=[instance],
                    options=options,
                    controller_callback=controllers.pop(alias),
                )

        return instance

    @classmethod
    def add_controller(
        cls,
        name: str,
        controller: Type[BaseFlowController],
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


class FlowControllerLiner(BaseFlowController):
    """A standard flow controller that runs tasks one after the other."""

    def yield_pipeline(self) -> Iterator[OptimizerTask]:
        yield from self.pipeline


class DoWhileController(ControllableController):
    """A flow controller that repeatedly run the entire pipeline until the condition is not met."""

    def yield_pipeline(self) -> Iterator[OptimizerTask]:
        max_iteration = self._options.get("max_iteration", 1000)
        for _ in range(max_iteration):
            yield from self.pipeline

            if not self.controller_callback(self.fenced_property_set):
                return

        raise PassManagerError("Maximum iteration reached. max_iteration=%i" % max_iteration)


class ConditionalController(ControllableController):
    """A flow controller runs the pipeline once when the condition is met."""

    def yield_pipeline(self) -> Iterator[OptimizerTask]:
        if self.controller_callback(self.fenced_property_set):
            yield from self.pipeline


FlowController.add_controller("condition", ConditionalController)
FlowController.add_controller("do_while", DoWhileController)
