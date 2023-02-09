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
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, Union, Optional

from qiskit.utils.deprecation import deprecate_exception

from .base_pass import BasePass
from .exceptions import PassManagerError
from .flow_controller import FlowController, PassSequence
from .propertyset import init_property_set, get_property_set


class BasePassRunner(ABC):
    """A stateful pass manager runner which is spawned by the pass manager run.

    The base pass runner must be agnostic to the pass manager IR and pass type.
    The flow controller is also agnostic to the IR and atomic passes
    as long as the passes are defined based upon the MetaPass metaclass.
    This "vail of ignorance" allows us to complete the flow control within
    the base pass runner, and each runner subclass only need to implement
    a mechanism to run an atomic pass and the data conversion logic across IRs/codes.

    A BasePassRunner subclass must be implemented for each combination of
    the pass manager IR, input code and target code.

    A flow controller provides custom interator to loop over the
    underlying passes, and the pass runner instance receives normalized controllers
    from the pass manager instance when created.
    All passes and controllers under the pass runner share
    the same thread local variable "property_set" implemented by the python contextvar,
    which is updated through the pass execution.

    The runner only takes a single input program and returns a single program.
    The pass manager IR must be consistent through the execution of registered passes.
    Note that input code and output code don't need to be identical type.
    """

    def __init_subclass__(cls, passmanager_error=None, **kwargs):
        # Temp fix for backward compatibility.
        # This method will be removed once we migrate to new exception PassManagerError.
        super().__init_subclass__(**kwargs)

        if passmanager_error is not None:
            import inspect

            deprecator = deprecate_exception(
                old_exception=passmanager_error,
                new_exception=PassManagerError,
                msg=f"{passmanager_error.__name__} exception is deprecated and will in future "
                "be replaced with PassManagerError exception.",
                category=PendingDeprecationWarning,
            )
            for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                if name.startswith("_"):
                    continue
                wrapped_method = deprecator(method)
                setattr(cls, name, wrapped_method)

    def __init__(self, max_iteration: int):
        """Initialize an empty PassManager object (with no passes scheduled).

        Args:
            max_iteration: The schedule looping iterates until the condition is met or until
                max_iteration is reached.
        """
        # the pass manager's schedule of passes, including any control-flow.
        # Populated via PassManager.append().
        self.working_list = []

        # passes already run that have not been invalidated
        self.valid_passes = set()

        # pass manager's overriding options for the passes it runs (for debugging)
        # This is no longer used -- Naoki Kanazawa (Qiskit/qiskit-terra/#9163)
        self.passmanager_options = {"max_iteration": max_iteration}

    def append(
        self,
        passes: PassSequence,
        **flow_controller_conditions: Callable,
    ):
        """Append a passes to the schedule of passes.

        Args:
            passes: A list of passes to be added to schedule.
            flow_controller_conditions: Dictionary of control flow plugins.
                This is going to be deprecated. Provide flow controller rather than
                a pass list with flow_controller_conditions.

        Raises:
            PassManagerError: When invalid flow condition is provided.
        """
        # TODO Remove this. Now standard input is only normalized flow controller.
        #  This lives only for unittest where it can take un-formatted pass list input.
        normalized_flow_controller = FlowController.controller_factory(
            passes=passes,
            options=self.passmanager_options,
            **flow_controller_conditions,
        )
        self.working_list.append(normalized_flow_controller)

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
    def _run_base_pass(
        self,
        pass_: BasePass,
        passmanager_ir: Any,
        options: Dict,
    ) -> Any:
        """Do a single base pass.

        Args:
            pass_: A base pass to run.
            passmanager_ir: Pass manager IR.
            options: PassManager options.

        Returns:
            Pass manager IR with optimization.
        """
        pass

    def _run_pass_generic(
        self,
        pass_sequence: Union[BasePass, FlowController],
        passmanager_ir: Any,
        options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Do either base pass or single flow controller.

        Args:
            pass_sequence: Base pass or flow controller to run.
            passmanager_ir: Pass manager IR.
            options: PassManager options.

        Returns:
            Pass manager IR with optimization.

        Raises:
            PassManagerError: When pass_sequence is not a valid class.
        """
        if isinstance(pass_sequence, BasePass):
            # Allow mutation of property set by pass execution.
            pass_sequence.property_set = get_property_set()
            passmanager_ir = self._run_base_pass(
                pass_=pass_sequence,
                passmanager_ir=passmanager_ir,
                options=options,
            )
            return passmanager_ir
        if isinstance(pass_sequence, FlowController):
            for pass_ in pass_sequence:
                passmanager_ir = self._run_pass_generic(
                    pass_sequence=pass_,
                    passmanager_ir=passmanager_ir,
                    options=pass_sequence.options,
                )
            return passmanager_ir
        raise PassManagerError(
            f"{pass_sequence.__class__} is not a valid base pass nor flow controller."
        )

    def run(self, in_program: Any) -> Any:
        """Run all the passes on an input program.

        Args:
            in_program: Input program to compile via all the registered passes.

        Returns:
            Compiled or optimized program.
        """
        # Create thread local propety set.
        init_property_set()

        passmanager_ir = self._to_passmanager_ir(in_program)
        del in_program

        for controller in self.working_list:
            passmanager_ir = self._run_pass_generic(
                pass_sequence=controller,
                passmanager_ir=passmanager_ir,
                options=self.passmanager_options,
            )

        return self._to_target(passmanager_ir)

    def _update_valid_passes(self, pass_):
        self.valid_passes.add(pass_)
        if not pass_.is_analysis_pass:  # Analysis passes preserve all
            self.valid_passes.intersection_update(set(pass_.preserves))
