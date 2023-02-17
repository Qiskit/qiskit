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

"""Qiskit pass manager for compilation and code optimization."""

# pylint: disable=ungrouped-imports, no-name-in-module
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional

from qiskit.utils.deprecation import deprecate_exception

from .base_pass import BasePass
from .exceptions import PassManagerError
from .flow_controller import FlowController
from .propertyset import init_property_set, get_property_set


class BasePassRunner(ABC):
    """Pass runner base class."""

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
        self.passmanager_options = {"max_iteration": max_iteration}

    def append(
        self,
        flow_controller: FlowController,
    ):
        """Append a flow controller to the schedule of controllers.

        Args:
            flow_controller: A normalized flow controller instance.
        """
        self.working_list.append(flow_controller)

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
    ) -> Any:
        """Do a single base pass.

        Args:
            pass_: A base pass to run.
            passmanager_ir: Pass manager IR.

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
            # First, do the requirements of this pass
            for required_pass in pass_sequence.requires:
                passmanager_ir = self._run_pass_generic(
                    pass_sequence=required_pass,
                    passmanager_ir=passmanager_ir,
                    options=options,
                )
            # Run the pass itself, if not already run
            if pass_sequence not in self.valid_passes:
                passmanager_ir = self._run_base_pass(
                    pass_=pass_sequence,
                    passmanager_ir=passmanager_ir,
                )
                self._update_valid_passes(pass_sequence)
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
        # Create thread local property set.
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
