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

# pylint: disable=ungrouped-imports, no-name-in-module
import sys
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Union, Sequence

from .base_pass import BasePass
from .exceptions import PassManagerError
from .fencedobjs import FencedPropertySet
from .flow_controller import FlowController, ConditionalController, DoWhileController
from .propertyset import PropertySet

if sys.version_info >= (3, 8):
    from functools import singledispatchmethod
else:
    from singledispatchmethod import singledispatchmethod


class BasePassRunner(ABC):
    """A stateful pass manager runner which is spawned by the pass manager run.

    The base pass runner must be agnostic to the pass manager IR and pass type.
    The flow controller is also agnostic to the IR and atomic passes
    as long as the passes are defined based upon the MetaPass metaclass.
    This "vail of ignorance" allows us to complete the flow control within
    the base pass runner, and each runner subclass only need to implement
    a mechanism to run an atomic pass and the data conversion logic across IR.

    Note that actual flow control logic is implemented by a generic method :meth:`_manage_flow`
    and it dispatches the algorithm based on the controller types.
    This allows not only conditional iteration but also more complicated flow control.

    A runner subclass must be implemented for each combination of
    the pass manager IR, input code and target code.

    The runner only takes a single input program and returns a single program.
    The pass manager IR must be consistent through the registered passes.
    Note that input code and output code don't need to be identical type.
    """

    def __init__(self, max_iteration: int):
        """Initialize an empty PassManager object (with no passes scheduled).

        Args:
            max_iteration: The schedule looping iterates until the condition is met or until
                max_iteration is reached.
        """
        # the pass manager's schedule of passes, including any control-flow.
        # Populated via PassManager.append().
        self.working_list = []

        # global property set is the context of the circuit held by the pass manager
        # as it runs through its scheduled passes. The flow controller
        # have read-only access (via the fenced_property_set).
        self.property_set = PropertySet()
        self.fenced_property_set = FencedPropertySet(self.property_set)

        # passes already run that have not been invalidated
        self.valid_passes = set()

        # pass manager's overriding options for the passes it runs (for debugging)
        self.passmanager_options = {"max_iteration": max_iteration}

    def append(
        self,
        passes: Union[Sequence[BasePass], FlowController],
        **flow_controller_conditions: Callable,
    ):
        """Append a passes to the schedule of passes.

        Args:
            passes: A list of passes to be added to schedule.
            flow_controller_conditions: See add_flow_controller(): Dictionary of
            control flow plugins. Default:

                * do_while (callable property_set -> boolean): The passes repeat until the
                  callable returns False.
                  Default: `lambda x: False # i.e. passes run once`

                * condition (callable property_set -> boolean): The passes run only if the
                  callable returns True.
                  Default: `lambda x: True # i.e. passes run`

        Raises:
            PassManagerError: When invalid flow condition is provided.
        """
        if isinstance(passes, ConditionalController) and not isinstance(passes.condition, partial):
            passes.condition = partial(passes.condition, self.fenced_property_set)

        if isinstance(passes, DoWhileController) and not isinstance(passes.do_while, partial):
            passes.do_while = partial(passes.do_while, self.fenced_property_set)

        if not isinstance(passes, FlowController):
            partial_controllers = {}
            for name, condition in flow_controller_conditions.items():
                if callable(condition):
                    partial_controllers[name] = partial(condition, self.fenced_property_set)
                else:
                    raise PassManagerError(f"The flow controller parameter {name} is not callable.")
            passes = FlowController.controller_factory(
                passes=passes,
                options=self.passmanager_options,
                **partial_controllers,
            )

        self.working_list.append(passes)

    @abstractmethod
    def _to_passmanager_ir(self, in_program):
        """Convert input program into pass manager IR.

        Args:
            in_program: Input program.

        Returns:
            Pass manager IR.
        """
        pass

    @abstractmethod
    def _to_target(self, passmanager_ir):
        """Convert pass manager IR into output program.

        Args:
            passmanager_ir: Pass manager IR after optimization.

        Returns:
            Output program.
        """
        pass

    @abstractmethod
    def _do_atomic_pass(self, pass_, passmanager_ir, options):
        """Do an atomic pass.

        Args:
            pass_: Pass to run.
            passmanager_ir: Pass manager IR.
            options: PassManager options.

        Returns:
            Pass manager IR with optimization.
        """
        pass

    @singledispatchmethod
    def _manage_flow(self, controller, passmanager_ir):
        # Do sinple iteration over passes. No conditional control.
        for sub_pass in controller:
            passmanager_ir = self._do_atomic_pass(
                pass_=sub_pass,
                passmanager_ir=passmanager_ir,
                options=controller.options,
            )
        return passmanager_ir

    @_manage_flow.register(ConditionalController)
    def _(self, controller, passmanager_ir):
        if not isinstance(controller.condition, partial):
            controller.condition = partial(controller.condition, self.fenced_property_set)

        # Run pass only when condition is satisfied.
        if controller.condition():
            for sub_pass in controller:
                passmanager_ir = self._do_atomic_pass(
                    pass_=sub_pass,
                    passmanager_ir=passmanager_ir,
                    options=controller.options,
                )
        return passmanager_ir

    @_manage_flow.register(DoWhileController)
    def _(self, controller, passmanager_ir):
        if not isinstance(controller.do_while, partial):
            controller.do_while = partial(controller.do_while, self.fenced_property_set)

        # Run pass until max iteration is reached or terminated by condition
        for _ in range(controller.max_iteration):
            for sub_pass in controller:
                passmanager_ir = self._do_atomic_pass(
                    pass_=sub_pass,
                    passmanager_ir=passmanager_ir,
                    options=controller.options,
                )
            if not controller.do_while():
                break
        return passmanager_ir

    # pylint: disable=missing-type-doc, missing-return-type-doc
    def run(self, in_program):
        """Run all the passes on an input program.

        Args:
            in_program: Input program to compile via all the registered passes.

        Returns:
            Compiled or optimized program.
        """
        passmanager_ir = self._to_passmanager_ir(in_program)
        del in_program

        for controller in self.working_list:
            passmanager_ir = self._manage_flow(controller, passmanager_ir)

        return self._to_target(passmanager_ir)

    def _update_valid_passes(self, pass_):
        self.valid_passes.add(pass_)
        if not pass_.is_analysis_pass:  # Analysis passes preserve all
            self.valid_passes.intersection_update(set(pass_.preserves))
