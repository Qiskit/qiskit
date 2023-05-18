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
from abc import ABC
from collections.abc import Callable, Sequence
from typing import Any
import logging
import dill
from qiskit.tools.parallel import parallel_map

from .passrunner import BasePassRunner
from .exceptions import PassManagerError
from .flow_controllers import FlowController, PassSequence

logger = logging.getLogger(__name__)


class BasePassManager(ABC):
    """Pass manager base class."""

    PASS_RUNNER = BasePassRunner

    def __init__(
        self,
        passes: PassSequence | None = None,
        max_iteration: int = 1000,
    ):
        """Initialize an empty pass manager object.

        Args:
            passes: A pass set to be added to the pass manager schedule.
            max_iteration: The maximum number of iterations the schedule will be looped if the
                condition is not met.
        """
        self._flow_controllers: list[FlowController] = []
        self.max_iteration = max_iteration

        if passes is not None:
            self.append(passes)

    def append(
        self,
        passes: PassSequence,
        **flow_controller_conditions: Callable,
    ) -> None:
        """Append a Pass Set to the schedule of passes.

        Args:
            passes: A set of passes (a pass set) to be added to schedule. A pass set is a list of
                    passes that are controlled by the same flow controller. If a single pass is
                    provided, the pass set will only have that pass a single element.
                    It is also possible to append a
                    :class:`~qiskit.transpiler.runningpassmanager.FlowController` instance and the
                    rest of the parameter will be ignored.
            flow_controller_conditions: Dictionary of control flow plugins. Default:

                * do_while (callable property_set -> boolean): The passes repeat until the
                  callable returns False.
                  Default: `lambda x: False # i.e. passes run once`

                * condition (callable property_set -> boolean): The passes run only if the
                  callable returns True.
                  Default: `lambda x: True # i.e. passes run`
        """
        normalized_flow_controller = FlowController.controller_factory(
            passes=passes,
            options={"max_iteration": self.max_iteration},
            **flow_controller_conditions,
        )
        self._flow_controllers.append(normalized_flow_controller)

    def replace(
        self,
        index: int,
        passes: PassSequence,
        **flow_controller_conditions: Any,
    ) -> None:
        """Replace a particular pass in the scheduler.

        Args:
            index: Pass index to replace, based on the position in passes().
            passes: A pass set (as defined in :py:func:`qiskit.transpiler.PassManager.append`)
                to be added to the pass manager schedule.
            flow_controller_conditions: control flow plugins.

        Raises:
            PassManagerError: if a pass in passes is not a proper pass or index not found.
        """
        normalized_flow_controller = FlowController.controller_factory(
            passes=passes,
            options={"max_iteration": self.max_iteration},
            **flow_controller_conditions,
        )
        try:
            self._flow_controllers[index] = normalized_flow_controller
        except IndexError as ex:
            raise PassManagerError(f"Index to replace {index} does not exists") from ex

    def remove(self, index: int) -> None:
        """Removes a particular pass in the scheduler.

        Args:
            index: Pass index to replace, based on the position in passes().

        Raises:
            PassManagerError: if the index is not found.
        """
        try:
            del self._flow_controllers[index]
        except IndexError as ex:
            raise PassManagerError(f"Index to replace {index} does not exists") from ex

    def __setitem__(self, index, item):
        self.replace(index, item)

    def __len__(self):
        return len(self._flow_controllers)

    def __getitem__(self, index):
        new_passmanager = self.__class__(max_iteration=self.max_iteration)
        new_passmanager._flow_controllers = [self._flow_controllers[index]]
        return new_passmanager

    def __add__(self, other):
        new_passmanager = self.__class__(max_iteration=self.max_iteration)
        new_passmanager._flow_controllers = self._flow_controllers
        if isinstance(other, self.__class__):
            new_passmanager._flow_controllers += other._flow_controllers
            return new_passmanager
        else:
            try:
                new_passmanager.append(other)
                return new_passmanager
            except PassManagerError as ex:
                raise TypeError(
                    "unsupported operand type + for %s and %s" % (self.__class__, other.__class__)
                ) from ex

    def run(
        self,
        in_programs: Any,
        callback: Callable | None = None,
        **metadata,
    ) -> Any:
        """Run all the passes on the specified ``circuits``.

        Args:
            in_programs: Input programs to transform via all the registered passes.
            callback: A callback function that will be called after each pass execution. The
                function will be called with 5 keyword arguments::

                    pass_ (Pass): the pass being run
                    passmanager_ir (Any): depending on pass manager subclass
                    time (float): the time to execute the pass
                    property_set (PropertySet): the property set
                    count (int): the index for the pass execution

                The exact arguments pass expose the internals of the pass
                manager and are subject to change as the pass manager internals
                change. If you intend to reuse a callback function over
                multiple releases be sure to check that the arguments being
                passed are the same.

                To use the callback feature you define a function that will
                take in kwargs dict and access the variables. For example::

                    def callback_func(**kwargs):
                        pass_ = kwargs['pass_']
                        dag = kwargs['dag']
                        time = kwargs['time']
                        property_set = kwargs['property_set']
                        count = kwargs['count']
                        ...

            metadata: Metadata which might be attached to output program.

        Returns:
            The transformed program(s).
        """
        if not self._flow_controllers and not metadata and callback is None:
            return in_programs

        # Create pass runner from normalized flow controllers
        # pylint: disable=abstract-class-instantiated
        pass_runner = self.PASS_RUNNER(self.max_iteration)
        for controller in self._flow_controllers:
            pass_runner.append(controller)

        is_list = True
        if isinstance(in_programs, self.PASS_RUNNER.IN_PROGRAM_TYPE):
            in_programs = [in_programs]
            is_list = False

        if len(in_programs) == 1:
            out_program = self._run_single_circuit(
                pass_runner=pass_runner,
                input_program=in_programs[0],
                callback=callback,
                **metadata,
            )
            if is_list:
                return [out_program]
            return out_program

        # TODO support for List(output_name) and List(callback)
        del metadata
        del callback

        return self._run_several_circuits(
            pass_runner=pass_runner,
            input_programs=in_programs,
        )

    def _run_single_circuit(
        self,
        pass_runner: BasePassRunner,
        input_program: Any,
        callback: Callable | None = None,
        **metadata,
    ) -> Any:
        return pass_runner.run(input_program, callback=callback, **metadata)

    def _run_several_circuits(
        self,
        pass_runner: BasePassRunner,
        input_programs: Sequence[Any],
    ) -> Any:
        # Pass runner may contain callable and we need to serialize through dill rather than pickle.
        # See https://github.com/Qiskit/qiskit-terra/pull/3290
        # Note that serialized object is deserialized as a different object.
        # Thus, we can resue the same runner without state collision, without building it per thread.
        serialized_runner = dill.dumps(pass_runner)
        return parallel_map(
            self._in_parallel, input_programs, task_kwargs={"runner_dill": serialized_runner}
        )

    @staticmethod
    def _in_parallel(
        in_program: Any,
        runner_dill: bytes = None,
    ) -> Any:
        pass_runner = dill.loads(runner_dill)
        return pass_runner.run(in_program)

    def to_flow_controller(self) -> FlowController:
        """Linearize this manager into a single :class:`.FlowController`, so that it can be nested
        inside another pass manager."""
        return FlowController.controller_factory(self._flow_controllers, {})
