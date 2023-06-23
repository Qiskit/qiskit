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

"""Pass runner to apply transformation on passmanager IR."""
from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from functools import partial
from collections.abc import Callable
from typing import Any

from .base_pass import GenericPass
from .exceptions import PassManagerError
from .flow_controllers import FlowController, ConditionalController, DoWhileController
from .propertyset import PropertySet

logger = logging.getLogger(__name__)

# NoneType is removed from types module in < Python3.10.
NoneType = type(None)


class BasePassRunner(ABC):
    """Pass runner base class."""

    IN_PROGRAM_TYPE = NoneType
    OUT_PROGRAM_TYPE = NoneType
    IR_TYPE = NoneType

    def __init__(self, max_iteration: int):
        """Initialize an empty pass runner object.

        Args:
            max_iteration: The schedule looping iterates until the condition is met or until
                max_iteration is reached.
        """
        self.callback = None
        self.count = None
        self.metadata = None

        # the pass manager's schedule of passes, including any control-flow.
        # Populated via PassManager.append().
        self.working_list = []

        # global property set is the context of the circuit held by the pass manager
        # as it runs through its scheduled passes. The flow controller
        # have read-only access (via the fenced_property_set).
        self.property_set = PropertySet()

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
        # We assume flow controller is already normalized.
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
        pass_: GenericPass,
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
        pass_sequence: GenericPass | FlowController,
        passmanager_ir: Any,
        options: dict[str, Any] | None = None,
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
            TypeError: When IR type changed during transformation.
        """
        if isinstance(pass_sequence, GenericPass):
            # First, do the requirements of this pass
            for required_pass in pass_sequence.requires:
                passmanager_ir = self._run_pass_generic(
                    pass_sequence=required_pass,
                    passmanager_ir=passmanager_ir,
                    options=options,
                )
            # Run the pass itself, if not already run
            if pass_sequence not in self.valid_passes:
                start_time = time.time()
                try:
                    passmanager_ir = self._run_base_pass(
                        pass_=pass_sequence,
                        passmanager_ir=passmanager_ir,
                    )
                finally:
                    run_time = time.time() - start_time
                    log_msg = f"Pass: {pass_sequence.name()} - {run_time * 1000:.5f} (ms)"
                    logger.info(log_msg)
                if self.callback:
                    self.callback(
                        pass_=pass_sequence,
                        passmanager_ir=passmanager_ir,
                        time=run_time,
                        property_set=self.property_set,
                        count=self.count,
                    )
                    self.count += 1
                self._update_valid_passes(pass_sequence)
            if not isinstance(passmanager_ir, self.IR_TYPE):
                raise TypeError(
                    f"A transformed object {passmanager_ir} is not valid IR in this pass manager. "
                    "Object representation type must be preserved during transformation. "
                    f"The pass {pass_sequence.name()} returns invalid object."
                )
            return passmanager_ir

        if isinstance(pass_sequence, FlowController):
            # This will be removed in followup PR. Code is temporary.
            fenced_property_set = getattr(self, "fenced_property_set")

            if isinstance(pass_sequence, ConditionalController) and not isinstance(
                pass_sequence.condition, partial
            ):
                pass_sequence.condition = partial(pass_sequence.condition, fenced_property_set)
            if isinstance(pass_sequence, DoWhileController) and not isinstance(
                pass_sequence.do_while, partial
            ):
                pass_sequence.do_while = partial(pass_sequence.do_while, fenced_property_set)
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

    def run(
        self,
        in_program: Any,
        callback: Callable | None = None,
        **metadata,
    ) -> Any:
        """Run all the passes on an input program.

        Args:
            in_program: Input program to compile via all the registered passes.
            callback: A callback function that will be called after each pass execution.
            **metadata: Metadata attached to the output program.

        Returns:
            Compiled or optimized program.

        Raises:
            TypeError: When input or output object is unexpected type.
        """
        if not isinstance(in_program, self.IN_PROGRAM_TYPE):
            raise TypeError(
                f"Input object {in_program} is not valid type for this pass manager. "
                f"This pass manager accepts {self.IN_PROGRAM_TYPE}."
            )

        if callback:
            self.callback = callback
            self.count = 0
        self.metadata = metadata

        passmanager_ir = self._to_passmanager_ir(in_program)
        del in_program

        for controller in self.working_list:
            passmanager_ir = self._run_pass_generic(
                pass_sequence=controller,
                passmanager_ir=passmanager_ir,
                options=self.passmanager_options,
            )
        out_program = self._to_target(passmanager_ir)

        if not isinstance(out_program, self.OUT_PROGRAM_TYPE):
            raise TypeError(
                f"Output object {out_program} is not valid type for this pass manager. "
                f"This pass manager must return {self.OUT_PROGRAM_TYPE}."
            )
        return out_program

    def _update_valid_passes(self, pass_):
        self.valid_passes.add(pass_)
