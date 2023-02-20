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

from abc import ABC, abstractmethod
from typing import List, Callable, Any

import dill
from qiskit.tools.parallel import parallel_map
from qiskit.utils.deprecation import deprecate_exception

from .base_pass_runner import BasePassRunner
from .exceptions import PassManagerError
from .flow_controller import FlowController, PassSequence
from .propertyset import get_property_set


class BasePassManager(ABC):
    """Pass manager base class."""

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

    def __init__(
        self,
        passes: PassSequence = None,
        max_iteration: int = 1000,
    ):
        """Initialize an empty `PassManager` object (with no passes scheduled).

        Args:
            passes: A pass set (as defined in :py:func:`qiskit.transpiler.PassManager.append`)
                to be added to the pass manager schedule.
            max_iteration: The maximum number of iterations the schedule will be looped if the
                condition is not met.
        """
        # the pass manager's schedule of passes, including any control-flow.
        # Populated via PassManager.append().

        self._flow_controllers = []
        self.max_iteration = max_iteration
        self.property_set = None

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
        else:
            try:
                new_passmanager.append(other)
            except PassManagerError as ex:
                raise TypeError(
                    f"unsupported operand type + for {self.__class__} and {other.__class__}"
                ) from ex

        return new_passmanager

    def run(
        self,
        in_programs: Any,
        **run_options: Any,
    ) -> Any:
        """Run all the passes on the specified ``circuits``.

        Args:
            in_programs: Input programs to transform via all the registered passes.
            run_options: Arbitrary run options supported by the pass runner.

        Returns:
            The transformed program(s).
        """
        pass_runner = self._create_running_passmanager()

        if len(in_programs) == 1:
            result = pass_runner.run(in_programs[0], **run_options)
            self.property_set = get_property_set()
            return result

        # Pass runner may contain callable and we need to serialize through dill rather than pickle.
        # See https://github.com/Qiskit/qiskit-terra/pull/3290
        # Note that serialized object is deserialized as a different object.
        # Thus, we can resue the same runner without state collision, without building it per thread.
        serialized_runner = dill.dumps(pass_runner)

        # TODO support for List(output_name) and List(callback)
        del run_options

        # TODO convert pass instance variables to context var and switch to asyncio.
        # In principle we can avoid inefficient copies with proper multi worker design.
        return parallel_map(
            BasePassManager._in_parallel,
            in_programs,
            task_kwargs={"serialized_pass_runner": serialized_runner},
        )

    @abstractmethod
    def _create_running_passmanager(self) -> BasePassRunner:
        pass

    @staticmethod
    def _in_parallel(
        in_program: Any,
        serialized_pass_runner: bytes = None,
    ) -> Any:
        """Task used by the parallel map tools from ``_run_several_circuits``."""
        pass_runner = dill.loads(serialized_pass_runner)
        return pass_runner.run(in_program)

    def draw(self, filename=None, style=None, raw=False):
        """Draw the pass manager.

        This function needs `pydot <https://github.com/erocarrera/pydot>`__, which in turn needs
        `Graphviz <https://www.graphviz.org/>`__ to be installed.

        Args:
            filename (str): file path to save image to.
            style (dict): keys are the pass classes and the values are the colors to make them. An
                example can be seen in the DEFAULT_STYLE. An ordered dict can be used to ensure
                a priority coloring when pass falls into multiple categories. Any values not
                included in the provided dict will be filled in from the default dict.
            raw (bool): If ``True``, save the raw Dot output instead of the image.

        Returns:
            Optional[PassManager]: an in-memory representation of the pass manager, or ``None``
            if no image was generated or `Pillow <https://pypi.org/project/Pillow/>`__
            is not installed.

        Raises:
            ImportError: when nxpd or pydot not installed.
        """
        from qiskit.visualization import pass_manager_drawer

        return pass_manager_drawer(self, filename=filename, style=style, raw=raw)

    @property
    def flow_controllers(self) -> List[FlowController]:
        """Return a list of flow controllers."""
        return self._flow_controllers

    def to_flow_controller(self) -> FlowController:
        """Linearize this manager into a single :class:`.FlowController`, so that it can be nested
        inside another :class:`.PassManager`."""
        return FlowController.controller_factory(self._flow_controllers, {})
