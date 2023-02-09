# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Qiskit pass manager for compilation and code optimization."""

from collections import OrderedDict
from functools import partial
from typing import Callable, Sequence, List, Dict, Union

from .base_pass import BasePass
from .exceptions import PassManagerError
from .propertyset import FuturePropertySet


class FlowController:
    """Base class for multiple types of working list.

    This class is a base class for multiple types of working list.
    The flow controller must be agnostic to the pass manager IR and pass types.
    """

    registered_controllers = OrderedDict()

    def __init__(
        self,
        passes: Sequence[Union[BasePass, "FlowController"]],
        options: Dict,
        **partial_controller: Callable,
    ):
        # Usage of this protected member is not clear. Can be removed?
        self._passes = passes
        self.passes = FlowController.controller_factory(passes, options, **partial_controller)
        self.options = options

    def __iter__(self):
        yield from self.passes

    def dump_passes(self):
        """Fetches the passes added to this flow controller.

        Returns:
             dict: {'options': self.options, 'passes': [passes], 'type': type(self)}
        """
        # TODO remove
        ret = {"options": self.options, "passes": [], "type": type(self)}
        for pass_ in self._passes:
            if isinstance(pass_, FlowController):
                ret["passes"].append(pass_.dump_passes())
            else:
                ret["passes"].append(pass_)
        return ret

    @classmethod
    def add_flow_controller(cls, name: str, controller: "FlowController"):
        """Adds a flow controller.

        Args:
            name (string): Name of the controller to add.
            controller (type(BaseFlowController)): The class implementing a flow controller.
        """
        cls.registered_controllers[name] = controller

    @classmethod
    def remove_flow_controller(cls, name: str):
        """Removes a flow controller.

        Args:
            name: Name of the controller to remove.

        Raises:
            KeyError: If the controller to remove was not registered.
        """
        if name not in cls.registered_controllers:
            raise KeyError("Flow controller not found: %s" % name)
        del cls.registered_controllers[name]

    @classmethod
    def controller_factory(
        cls,
        passes: Sequence[Union[BasePass, "FlowController"]],
        options: Dict,
        **partial_controller: Callable,
    ) -> "FlowController":
        """Constructs a flow controller based on the partially evaluated controller arguments.

        Args:
            passes: passes to add to the flow controller.
            options: PassManager options.
            **partial_controller: Partially evaluated controller arguments in the form
                `{name:partial}`

        Raises:
            PassManagerError: When partial_controller is not well-formed.
            PassManagerError: Partial controller is not registered.

        Returns:
            FlowController: A FlowController instance.
        """
        passes = _normalize_passes_generic(passes)

        if None in partial_controller.values():
            raise PassManagerError("The controller needs a condition.")

        if partial_controller:
            # Validate controllers. Make sure context property set is tied to callables.
            for key, value in partial_controller.items():
                if callable(value) and not isinstance(value, partial):
                    partial_controller[key] = partial(value, FuturePropertySet())
            for tag, controller_cls in cls.registered_controllers.items():
                if tag in partial_controller:
                    return controller_cls(passes, options, **partial_controller)
            raise PassManagerError(f"The controllers for {partial_controller} are not registered")
        return FlowControllerLinear(passes, options)


class FlowControllerLinear(FlowController):
    """The basic controller runs the passes one after the other."""

    def __init__(self, passes, options):  # pylint: disable=super-init-not-called
        self.passes = self._passes = passes
        self.options = options


class DoWhileController(FlowController):
    """Implements a set of passes in a do-while loop."""

    def __init__(self, passes, options=None, do_while=None, **partial_controller):
        if not callable(do_while):
            raise PassManagerError("The flow controller parameter 'do_while' is not callable.")
        self.do_while = do_while
        self.max_iteration = options["max_iteration"] if options else 1000
        super().__init__(passes, options, **partial_controller)

    def __iter__(self):
        for _ in range(self.max_iteration):
            yield from self.passes

            if not self.do_while():
                return
        raise PassManagerError(f"Maximum iteration reached. max_iteration={self.max_iteration}")


class ConditionalController(FlowController):
    """Implements a set of passes under a certain condition."""

    def __init__(self, passes, options=None, condition=None, **partial_controller):
        if not callable(condition):
            raise PassManagerError("The flow controller parameter 'condition' is not callable.")
        self.condition = condition
        super().__init__(passes, options, **partial_controller)

    def __iter__(self):
        if self.condition():
            yield from self.passes


# Alias to the union of base pass and flow controller
PassSequence = Union[Union[BasePass, FlowController], List[Union[BasePass, FlowController]]]


def _bind_context_property(controller: FlowController) -> FlowController:
    """A helper function to make sure all callables take a property set.

    Args:
        controller: Flow controller instance to investigate.

    Returns:
        A flow controller tied to the property set.
    """
    excludes = ["_passes", "passes", "options"]

    # Hard-code built-in controllers since callable attribute name is known.
    if isinstance(controller, FlowControllerLinear):
        return controller
    if isinstance(controller, ConditionalController):
        if not isinstance(controller.condition, partial):
            controller.condition = partial(controller.condition, FuturePropertySet())
        return controller
    if isinstance(controller, DoWhileController):
        if not isinstance(controller.do_while, partial):
            controller.do_while = partial(controller.do_while, FuturePropertySet())
        return controller
    # Investigate all members when controller is not built-in.
    for attr_name, value in vars(controller).items():
        if attr_name in excludes:
            continue
        if callable(value) and not isinstance(value, partial):
            setattr(controller, attr_name, partial(value, FuturePropertySet()))
    return controller


def _normalize_passes_generic(passes: PassSequence) -> PassSequence:
    """A helper function to normalize passes.

    Args:
        passes: Passes to normalize.

    Returns:
        Normalized pass.

    Raises:
        TypeError: When invalid pass is provided.
    """
    if isinstance(passes, BasePass):
        # Base pass can go as-is.
        return passes
    if isinstance(passes, FlowController):
        # Controller that may take a callable requiring property set.
        normalized_inset_passes = _normalize_passes_generic(passes.passes)
        passes.passes = normalized_inset_passes
        return _bind_context_property(passes)
    if isinstance(passes, (list, tuple)):
        # Sequence of base passes or flow controllers.
        return list(map(_normalize_passes_generic, passes))
    raise TypeError(f"{passes.__class__} is not a valid BasePass of FlowController instance.")


# Default controllers
FlowController.add_flow_controller("condition", ConditionalController)
FlowController.add_flow_controller("do_while", DoWhileController)
