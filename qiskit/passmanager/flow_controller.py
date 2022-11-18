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
from typing import Callable, Sequence, Dict

from .base_pass import BasePass
from .exceptions import PassManagerError


class FlowController:
    """Base class for multiple types of working list.

    This class is a base class for multiple types of working list.
    The flow controller must be agnostic to the pass manager IR and pass types.
    """

    registered_controllers = OrderedDict()

    def __init__(
        self,
        passes: Sequence[BasePass],
        options: Dict,
        **partial_controller: Callable,
    ):
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
        passes: Sequence[BasePass],
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
        if None in partial_controller.values():
            raise PassManagerError("The controller needs a condition.")

        if partial_controller:
            for registered_controller in cls.registered_controllers.keys():
                if registered_controller in partial_controller:
                    return cls.registered_controllers[registered_controller](
                        passes, options, **partial_controller
                    )
            raise PassManagerError("The controllers for %s are not registered" % partial_controller)

        return FlowControllerLinear(passes, options)


class FlowControllerLinear(FlowController):
    """The basic controller runs the passes one after the other."""

    def __init__(self, passes, options):  # pylint: disable=super-init-not-called
        self.passes = self._passes = passes
        self.options = options


class DoWhileController(FlowController):
    """Implements a set of passes in a do-while loop."""

    def __init__(self, passes, options=None, do_while=None, **partial_controller):
        self.do_while = do_while
        self.max_iteration = options["max_iteration"] if options else 1000
        super().__init__(passes, options, **partial_controller)


class ConditionalController(FlowController):
    """Implements a set of passes under a certain condition."""

    def __init__(self, passes, options=None, condition=None, **partial_controller):
        self.condition = condition
        super().__init__(passes, options, **partial_controller)


# Default controllers
FlowController.add_flow_controller("condition", ConditionalController)
FlowController.add_flow_controller("do_while", DoWhileController)
