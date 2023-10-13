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
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import dill

from qiskit.tools.parallel import parallel_map
from .base_tasks import Task, BaseController, PassManagerIR
from .exceptions import PassManagerError
from .flow_controllers import FlowControllerLinear, FlowController
from .propertyset import PropertySet

logger = logging.getLogger(__name__)


class BasePassManager(ABC):
    """Pass manager base class."""

    def __init__(
        self,
        tasks: list[Task] = None,
        max_iteration: int = 1000,
    ):
        """Initialize an empty pass manager object.

        Args:
            tasks: A pass set to be added to the pass manager schedule.
            max_iteration: The maximum number of iterations the schedule will be looped if the
                condition is not met.
        """
        self._flow_controller = FlowControllerLinear()
        self.max_iteration = max_iteration
        self.property_set = PropertySet()

        # For backward compatibility.
        # This wraps pass list with a linear flow controller.
        if tasks is not None:
            self.append(tasks)

    def append(
        self,
        tasks: Task | list[Task],
        **flow_controller_conditions: Callable[[PropertySet], bool],
    ) -> None:
        """Append tasks to the schedule of passes.

        Args:
            tasks: A set of pass manager tasks to be added to schedule. When multiple
                tasks are provided, tasks are grouped together as a single flow controller.
            flow_controller_conditions: Dictionary of control flow plugins.
                Following built-in controllers are available by default:

                * do_while: The passes repeat until the callable returns False.
                * condition: The passes run only if the callable returns True.
        """
        if flow_controller_conditions:
            tasks = _legacy_build_flow_controller(
                tasks,
                options={"max_iteration": self.max_iteration},
                **flow_controller_conditions,
            )
        if isinstance(tasks, Sequence):
            tasks = FlowControllerLinear(tasks)
        self._flow_controller.tasks += (tasks, )

    def replace(
        self,
        index: int,
        tasks: Task | list[Task],
        **flow_controller_conditions: Any,
    ) -> None:
        """Replace a particular pass in the scheduler.

        Args:
            index: Pass index to replace, based on the position in passes().
            tasks: A set of pass manager tasks to be added to schedule. When multiple
                tasks are provided, tasks are grouped together as a single flow controller.
            flow_controller_conditions: Dictionary of control flow plugins.
                See :meth:`~.BasePassManager.append` for details.

        Raises:
            PassManagerError: If the index is not found.
        """
        if flow_controller_conditions:
            tasks = _legacy_build_flow_controller(
                tasks,
                options={"max_iteration": self.max_iteration},
                **flow_controller_conditions,
            )
        if isinstance(tasks, Sequence):
            tasks = FlowControllerLinear(tasks)
        try:
            new_tasks = list(self._flow_controller.tasks)
            new_tasks[index] = tasks
            self._flow_controller.tasks = tuple(new_tasks)
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
            new_tasks = list(self._flow_controller.tasks)
            del new_tasks[index]
            self._flow_controller.tasks = new_tasks
        except IndexError as ex:
            raise PassManagerError(f"Index to replace {index} does not exists") from ex

    def __setitem__(self, index, item):
        self.replace(index, item)

    def __len__(self):
        return len(self._flow_controller.tasks)

    def __getitem__(self, index):
        new_passmanager = self.__class__(max_iteration=self.max_iteration)
        new_controller = FlowControllerLinear(self._flow_controller.tasks[index])
        new_passmanager._flow_controller = new_controller
        return new_passmanager

    def __add__(self, other):
        new_passmanager = self.__class__(max_iteration=self.max_iteration)
        new_passmanager._flow_controller = self._flow_controller
        if isinstance(other, self.__class__):
            new_passmanager._flow_controller.tasks += other._flow_controller.tasks
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
    ) -> PassManagerIR:
        pass

    @abstractmethod
    def _passmanager_backend(
        self,
        passmanager_ir: PassManagerIR,
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
                    count (int): the index for the pass execution

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
                        count = kwargs['count']
                        ...

            kwargs: Arbitrary arguments passed to the compiler frontend and backend.

        Returns:
            The transformed program(s).
        """
        if not self._flow_controller.tasks and not kwargs and callback is None:
            return in_programs

        is_list = True
        if not isinstance(in_programs, Sequence):
            in_programs = [in_programs]
            is_list = False

        if len(in_programs) == 1:
            out_program = _run_workflow(
                program=in_programs[0],
                pass_manager=self,
                callback=callback,
                **kwargs,
            )
            if is_list:
                return [out_program]
            return out_program

        del callback
        del kwargs

        # Pass manager may contain callable and we need to serialize through dill rather than pickle.
        # See https://github.com/Qiskit/qiskit-terra/pull/3290
        # Note that serialized object is deserialized as a different object.
        # Thus, we can resue the same manager without state collision, without building it per thread.
        return parallel_map(
            _run_workflow_in_new_process,
            values=in_programs,
            task_kwargs={"pass_manager_bin": dill.dumps(self)},
        )

    def to_flow_controller(self) -> FlowControllerLinear:
        """Linearize this manager into a single :class:`.FlowControllerLiner`,
        so that it can be nested inside another pass manager.

        Returns:
            A linearized pass manager.
        """
        return self._flow_controller


def _run_workflow(
    program: Any,
    pass_manager: BasePassManager,
    **kwargs,
) -> Any:
    """Run single program optimization with a pass manager.

    Args:
        program: Arbitrary program to optimize.
        pass_manager: Pass manager with scheduled passes.
        **kwargs: Keyword arguments for IR conversion.

    Returns:
        Optimized program.
    """
    flow_controller = pass_manager.to_flow_controller()

    passmanager_ir = pass_manager._passmanager_frontend(input_program=program, **kwargs)
    passmanager_ir = flow_controller.execute(
        passmanager_ir=passmanager_ir,
        property_set=pass_manager.property_set,
        callback=kwargs.get("callback", None),
    )
    out_program = pass_manager._passmanager_backend(passmanager_ir, **kwargs)

    return out_program


def _run_workflow_in_new_process(
    program: Any,
    pass_manager_bin: bytes,
) -> Any:
    """Run single program optimization in new process.

    Args:
        program: Arbitrary program to optimize.
        pass_manager_bin: Binary of the pass manager with scheduled passes.

    Returns:
          Optimized program.
    """
    return _run_workflow(
        program=program,
        pass_manager=dill.loads(pass_manager_bin),
    )


def _legacy_build_flow_controller(
    tasks: list[Task],
    options: dict[str, Any],
    **flow_controller_conditions,
) -> BaseController:
    """A legacy method to build flow controller with keyword arguments.

    Args:
        tasks: A list of tasks fed into custom flow controllers.
        options: Option for flow controllers.
        flow_controller_conditions: Callables keyed on the alias of the flow controller.

    Returns:
        A built controller.
    """
    warnings.warn(
        "Building a flow controller with keyword arguments is going to be deprecated. "
        "Custom controllers must be explicitly instantiated and appended to the task list.",
        PendingDeprecationWarning,
    )

    # Alias in higher hierarchy becomes outer controller.
    for alias in FlowController.hierarchy[::-1]:
        if alias not in flow_controller_conditions:
            continue
        class_type = FlowController.registered_controllers[alias]
        init_kwargs = {
            "options": options,
            alias: flow_controller_conditions.pop(alias),
        }
        tasks = class_type(tasks, **init_kwargs)
    return tasks
