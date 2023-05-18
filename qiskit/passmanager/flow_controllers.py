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

"""Pass flow controllers to provide pass iterator conditioned on the property set."""
from __future__ import annotations
from collections import OrderedDict
from collections.abc import Sequence, Callable
from functools import partial
from typing import List, Union, Any
import logging

from .base_pass import GenericPass
from .propertyset import FuturePropertySet
from .exceptions import PassManagerError

logger = logging.getLogger(__name__)


class FlowController:
    """Base class for multiple types of working list.

    This class is a base class for multiple types of working list. When you iterate on it, it
    returns the next pass to run.
    """

    registered_controllers = OrderedDict()

    def __init__(
        self,
        passes: Sequence[GenericPass | "FlowController"],
        options: dict,
        **partial_controller: Callable,
    ):
        """Create new flow controller.

        Args:
            passes: passes to add to the flow controller. This must be normalized.
            options: PassManager options.
            **partial_controller: Partially evaluated callables representing
                a condition to invoke flow controllers. This dictionary
                must be keyed on the registered controller name.
        """
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
    def add_flow_controller(cls, name, controller):
        """Adds a flow controller.

        Args:
            name (string): Name of the controller to add.
            controller (type(FlowController)): The class implementing a flow controller.
        """
        cls.registered_controllers[name] = controller

    @classmethod
    def remove_flow_controller(cls, name):
        """Removes a flow controller.

        Args:
            name (string): Name of the controller to remove.

        Raises:
            KeyError: If the controller to remove was not registered.
        """
        if name not in cls.registered_controllers:
            raise KeyError("Flow controller not found: %s" % name)
        del cls.registered_controllers[name]

    @classmethod
    def controller_factory(
        cls,
        passes: Sequence[GenericPass | "FlowController"],
        options: dict,
        **partial_controller,
    ):
        """Constructs a flow controller based on the partially evaluated controller arguments.

        Args:
            passes: passes to add to the flow controller.
            options: PassManager options.
            **partial_controller: Partially evaluated callables representing
                a condition to invoke flow controllers. This dictionary
                must be keyed on the registered controller name.
                Callable may only take property set, and if the property set is unbound,
                the factory method implicitly binds :class:`.FuturePropertySet`.
                When multiple controllers and conditions are provided, this recursively
                initializes controllers according to the priority defined by
                the key order of the :attr:`FlowController.registered_controllers`.

        Raises:
            PassManagerError: When partial_controller is not well-formed.

        Returns:
            FlowController: A FlowController instance.
        """
        if not _is_passes_sequence(passes):
            raise PassManagerError(
                f"{passes.__class__} is not a valid BasePass of FlowController instance."
            )
        if None in partial_controller.values():
            raise PassManagerError("The controller needs a condition.")

        if partial_controller:
            for key, value in partial_controller.items():
                if callable(value) and not isinstance(value, partial):
                    partial_controller[key] = partial(value, FuturePropertySet())
            for label, controller_cls in cls.registered_controllers.items():
                if label in partial_controller:
                    return controller_cls(passes, options, **partial_controller)
            raise PassManagerError(f"The controllers for {partial_controller} are not registered")
        return FlowControllerLinear(passes, options)


def _is_passes_sequence(passes: Any) -> bool:
    """Check if input is valid pass sequence.

    Valid pass sequence representation are:

    * BasePass,
    * FlowController,
    * Sequence of BasePass or FlowController.

    Note that nested sequence is not a valid pass sequence.
    FlowController.passes must be validated in constructor.
    """
    if isinstance(passes, (GenericPass, FlowController)):
        return True
    for inner_pass in passes:
        if not isinstance(inner_pass, (GenericPass, FlowController)):
            return False
    return True


class FlowControllerLinear(FlowController):
    """The basic controller runs the passes one after the other."""

    def __init__(self, passes, options):  # pylint: disable=super-init-not-called
        self.passes = self._passes = passes
        self.options = options


class DoWhileController(FlowController):
    """Implements a set of passes in a do-while loop."""

    def __init__(
        self,
        passes: PassSequence,
        options: dict = None,
        do_while: Callable = None,
        **partial_controller: Callable,
    ):
        if not callable(do_while):
            raise PassManagerError("The flow controller parameter do_while is not callable.")
        if not isinstance(do_while, partial):
            do_while = partial(do_while, FuturePropertySet())
        self.do_while = do_while
        self.max_iteration = options["max_iteration"] if options else 1000
        super().__init__(passes, options, **partial_controller)

    def __iter__(self):
        for _ in range(self.max_iteration):
            yield from self.passes

            if not self.do_while():
                return

        raise PassManagerError("Maximum iteration reached. max_iteration=%i" % self.max_iteration)


class ConditionalController(FlowController):
    """Implements a set of passes under a certain condition."""

    def __init__(
        self,
        passes: PassSequence,
        options: dict = None,
        condition: Callable = None,
        **partial_controller: Callable,
    ):
        if not callable(condition):
            raise PassManagerError("The flow controller parameter condition is not callable.")
        if not isinstance(condition, partial):
            condition = partial(condition, FuturePropertySet())
        self.condition = condition
        super().__init__(passes, options, **partial_controller)

    def __iter__(self):
        if self.condition():
            yield from self.passes


# Alias to a sequence of all kind of pass elements
PassSequence = Union[Union[GenericPass, FlowController], List[Union[GenericPass, FlowController]]]

# Default controllers
FlowController.add_flow_controller("condition", ConditionalController)
FlowController.add_flow_controller("do_while", DoWhileController)
