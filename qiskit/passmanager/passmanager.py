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

"""Manager for a set of Passes and their scheduling during transpilation."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import dill

from qiskit.tools.parallel import parallel_map
from .base_optimization_tasks import OptimizerTask
from .exceptions import PassManagerError
from .flow_controllers import FlowControllerLiner, FlowController
from .propertyset import PropertySet

logger = logging.getLogger(__name__)


class BasePassManager(ABC):
    """Pass manager base class."""

    def __init__(
        self,
        passes: list[OptimizerTask] = None,
        max_iteration: int = 1000,
    ):
        """Initialize an empty pass manager object.

        Args:
            passes: A pass set to be added to the pass manager schedule.
            max_iteration: The maximum number of iterations the schedule will be looped if the
                condition is not met.
        """
        self._flow_controller = FlowControllerLiner()
        self.max_iteration = max_iteration
        self.property_set = PropertySet()

        if passes is not None:
            self.append(passes)

    def append(
        self,
        passes: OptimizerTask | list[OptimizerTask],
        **flow_controller_conditions: Callable[[PropertySet], bool],
    ) -> None:
        """Append a Pass Set to the schedule of passes.

        Args:
            passes: A set of passes (a pass set) to be added to schedule. A pass set is a list of
                passes that are controlled by the same flow controller. If a single pass is
                provided, the pass set will only have that pass a single element.
                It is also possible to append a :class:`.BaseFlowController` instance and
                the rest of the parameter will be ignored.
            flow_controller_conditions: Dictionary of control flow plugins.
                Following built-in controllers are available by default:

                * do_while: The passes repeat until the callable returns False.
                * condition: The passes run only if the callable returns True.
        """
        normalized_controller = FlowController(
            passes=passes,
            options={"max_iteration": self.max_iteration},
            **flow_controller_conditions,
        )
        self._flow_controller.pipeline.append(normalized_controller)

    def replace(
        self,
        index: int,
        passes: OptimizerTask | list[OptimizerTask],
        **flow_controller_conditions: Any,
    ) -> None:
        """Replace a particular pass in the scheduler.

        Args:
            index: Pass index to replace, based on the position in passes().
            passes: A pass set to be added to the pass manager schedule.
            flow_controller_conditions: Dictionary of control flow plugins.
                See :meth:`~.BasePassManager.append` for details.

        Raises:
            PassManagerError: If the index is not found.
        """
        normalized_controller = FlowController(
            passes=passes,
            options={"max_iteration": self.max_iteration},
            **flow_controller_conditions,
        )
        try:
            self._flow_controller.pipeline[index] = normalized_controller
        except IndexError as ex:
            raise PassManagerError(f"Index to replace {index} does not exists") from ex

    def remove(self, index: int) -> None:
        """Removes a particular pass in the scheduler.

        Args:
            index: Pass index to remove, based on the position in passes().

        Raises:
            PassManagerError: If the index is not found.
        """
        try:
            del self._flow_controller.pipeline[index]
        except IndexError as ex:
            raise PassManagerError(f"Index to replace {index} does not exists") from ex

    def __setitem__(self, index, item):
        self.replace(index, item)

    def __len__(self):
        return len(self._flow_controller.pipeline)

    def __getitem__(self, index):
        new_passmanager = self.__class__(max_iteration=self.max_iteration)
        new_controller = FlowControllerLiner([self._flow_controller.pipeline[index]])
        new_passmanager._flow_controller = new_controller
        return new_passmanager

    def __add__(self, other):
        new_passmanager = self.__class__(max_iteration=self.max_iteration)
        new_passmanager._flow_controller = self._flow_controller
        if isinstance(other, self.__class__):
            new_passmanager._flow_controller.pipeline += other._flow_controller.pipeline
            return new_passmanager
        else:
            try:
                new_passmanager.append(other)
                return new_passmanager
            except PassManagerError as ex:
                raise TypeError(
                    "unsupported operand type + for %s and %s" % (self.__class__, other.__class__)
                ) from ex

    @abstractmethod
    def _passmanager_frontend(
        self,
        input_program: Any,
        **kwargs,
    ) -> Any:
        pass

    @abstractmethod
    def _passmanager_backend(
        self,
        passmanager_ir: Any,
        **kwargs,
    ) -> Any:
        pass

    def run(
        self,
        in_programs: Any,
        callback: Callable = None,
        **kwargs,
    ) -> Any:
        """Run all the passes on the specified ``circuits``.

        Args:
            in_programs: Input programs to transform via all the registered passes.
            callback: A callback function that will be called after each pass execution. The
                function will be called with 4 keyword arguments::

                    task (GenericPass): the pass being run
                    passmanager_ir (Any): depending on pass manager subclass
                    property_set (PropertySet): the property set
                    running_time (float): the time to execute the pass

                The exact arguments pass expose the internals of the pass
                manager and are subject to change as the pass manager internals
                change. If you intend to reuse a callback function over
                multiple releases be sure to check that the arguments being
                passed are the same.

                To use the callback feature you define a function that will
                take in kwargs dict and access the variables. For example::

                    def callback_func(**kwargs):
                        task = kwargs['task']
                        passmanager_ir = kwargs['passmanager_ir']
                        property_set = kwargs['property_set']
                        running_time = kwargs['running_time']
                        ...

            kwargs: Arbitrary arguments passed to the compiler frontend and backend.

        Returns:
            The transformed program(s).
        """
        if not self._flow_controller.pipeline and not kwargs and callback is None:
            return in_programs

        if callback:
            self._flow_controller.callback = callback

        is_list = True
        if not isinstance(in_programs, Sequence):
            in_programs = [in_programs]
            is_list = False

        if len(in_programs) == 1:
            out_program = self._run_workflow(self, in_programs[0], **kwargs)
            if is_list:
                return [out_program]
            return out_program

        # Pass manager may contain callable and we need to serialize through dill rather than pickle.
        # See https://github.com/Qiskit/qiskit-terra/pull/3290
        # Note that serialized object is deserialized as a different object.
        # Thus, we can resue the same manager without state collision, without building it per thread.
        return parallel_map(
            lambda prog, pm_dill, **task_kwargs: self._run_workflow(
                pass_manager=dill.loads(pm_dill),
                program=prog,
                **task_kwargs,
            ),
            values=in_programs,
            task_kwargs={"pm_dill": dill.dumps(self)},
        )

    @staticmethod
    def _run_workflow(
        pass_manager: BasePassManager,
        program: Any,
        **kwargs,
    ) -> Any:
        flow_controller = pass_manager.to_flow_controller()

        passmanager_ir = pass_manager._passmanager_frontend(input_program=program, **kwargs)
        passmanager_ir = flow_controller.execute(passmanager_ir, pass_manager.property_set)
        out_program = pass_manager._passmanager_backend(passmanager_ir, **kwargs)

        return out_program

    def to_flow_controller(self) -> FlowControllerLiner:
        """Linearize this manager into a single :class:`.FlowControllerLiner`,
        so that it can be nested inside another pass manager.

        Returns:
            A linearized pass manager.
        """
        return self._flow_controller
