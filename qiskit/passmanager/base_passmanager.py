# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Manager for a set of Passes and their scheduling during transpilation."""

import sys
from abc import ABC, abstractmethod
from typing import Union, List, Callable, Dict, Any

import dill
from qiskit.tools.parallel import parallel_map

from .base_pass import BasePass
from .base_pass_runner import BasePassRunner
from .exceptions import PassManagerError
from .flow_controller import FlowController

if sys.version_info >= (3, 8):
    from functools import singledispatchmethod  # pylint: disable=no-name-in-module
else:
    from singledispatchmethod import singledispatchmethod


class BasePassManager(ABC):
    """Base class of pass manager.

    This base class must be aware of input code, output code and base pass type,
    and it must be responsible for the type check to guarantee that the all passes
    work with the targeting pass manager IR,
    and the IR is generatable from the input code.

    This also does thread management for the multiple input programs for performance,
    and thus a suclass only need to generate proper pass runner depending on the
    expected pass manager IR, which must be supplied by :meth:`._create_running_passmanager`.
    """

    PASS_TYPE = BasePass
    """Expected pass base class of this pass manager."""

    INPUT_TYPE = object
    """Expected input program type."""

    TARGET_TYPE = object
    """Expected output program type."""

    def __init__(
        self,
        passes: Union[PASS_TYPE, List[PASS_TYPE]] = None,
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

        self._pass_sets = []
        if passes is not None:
            self.append(passes)
        self.max_iteration = max_iteration
        self.property_set = None

    def append(
        self,
        passes: Union[FlowController, List[BasePass]],
        max_iteration: int = None,
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
            max_iteration: max number of iterations of passes.
            flow_controller_conditions: control flow plugins.
        """
        if max_iteration:
            # TODO remove this argument from append
            self.max_iteration = max_iteration

        if isinstance(passes, self.PASS_TYPE):
            passes = [passes]
        else:
            passes = self._normalize_passes(passes)

        self._pass_sets.append(
            {
                "passes": passes,
                "flow_controllers": flow_controller_conditions,
            }
        )

    def replace(
        self,
        index: int,
        passes: Any,
        max_iteration: int = None,
        **flow_controller_conditions: Any,
    ) -> None:
        """Replace a particular pass in the scheduler.

        Args:
            index: Pass index to replace, based on the position in passes().
            passes: A pass set (as defined in :py:func:`qiskit.transpiler.PassManager.append`)
                to be added to the pass manager schedule.
            max_iteration: max number of iterations of passes.
            flow_controller_conditions: control flow plugins.

        Raises:
            PassManagerError: if a pass in passes is not a proper pass or index not found.
        """
        if max_iteration:
            # TODO remove this argument from append
            self.max_iteration = max_iteration

        if isinstance(passes, self.PASS_TYPE):
            passes = [passes]
        else:
            passes = self._normalize_passes(passes)

        try:
            self._pass_sets[index] = {
                "passes": passes,
                "flow_controllers": flow_controller_conditions,
            }
        except IndexError as ex:
            raise PassManagerError(f"Index to replace {index} does not exists") from ex

    @singledispatchmethod
    def _normalize_passes(self, passes):
        if not isinstance(passes, self.PASS_TYPE):
            raise PassManagerError(
                f"The pass type {type(passes)} is not supported by {self.__class__.__name__}."
            )
        return passes

    @_normalize_passes.register(list)
    def _(self, passes):
        out_passes = []
        for pass_ in passes:
            out_passes.append(self._normalize_passes(pass_))
        return out_passes

    @_normalize_passes.register(FlowController)
    def _(self, passes):
        passes.passes = self._normalize_passes(passes.passes)
        return passes

    def remove(self, index: int) -> None:
        """Removes a particular pass in the scheduler.

        Args:
            index: Pass index to replace, based on the position in passes().

        Raises:
            PassManagerError: if the index is not found.
        """
        try:
            del self._pass_sets[index]
        except IndexError as ex:
            raise PassManagerError(f"Index to replace {index} does not exists") from ex

    def _new_instance(self):
        """Get new instance of subclass."""
        return self.__class__(max_iteration=self.max_iteration)

    def __setitem__(self, index, item):
        self.replace(index, item)

    def __len__(self):
        return len(self._pass_sets)

    def __getitem__(self, index):
        new_passmanager = self._new_instance()
        _pass_sets = self._pass_sets[index]
        if isinstance(_pass_sets, dict):
            _pass_sets = [_pass_sets]
        new_passmanager._pass_sets = _pass_sets
        return new_passmanager

    def __add__(self, other):
        if isinstance(other, self.__class__):
            new_passmanager = self._new_instance()
            new_passmanager._pass_sets = self._pass_sets + other._pass_sets
            return new_passmanager
        else:
            try:
                new_passmanager = self._new_instance()
                new_passmanager._pass_sets += self._pass_sets
                new_passmanager.append(other)
                return new_passmanager
            except PassManagerError as ex:
                raise TypeError(
                    f"unsupported operand type + for {self.__class__} and {other.__class__}"
                ) from ex

    def run(
        self,
        in_programs: Union[INPUT_TYPE, List[INPUT_TYPE]],
        **run_options: Any,
    ) -> Union[TARGET_TYPE, List[TARGET_TYPE]]:
        """Run all the passes on the specified ``circuits``.

        Args:
            in_programs: Input program(s) to transform via all the registered passes.
            run_options: Arbitrary run options supported by the pass runner.

        Returns:
            The transformed program(s).

        Raises:
            PassManagerError: When the type of input programs is not consistent and
                not supported by the pass manager.
        """
        try:
            iter(in_programs)
        except TypeError:
            in_programs = [in_programs]

        # Validate input programs. Mixed input is not allowd.
        for in_program in in_programs:
            if not isinstance(in_program, self.INPUT_TYPE):
                raise PassManagerError(f"Invalid input program type {type(in_program)}.")

        pass_runner = self._create_running_passmanager()

        if len(in_programs) == 1:
            result = pass_runner.run(in_programs[0], **run_options)
            self.property_set = pass_runner.property_set
            return result

        # Pass runner may contain callable and we need to serialize through dill rather than pickle.
        # See https://github.com/Qiskit/qiskit-terra/pull/3290
        # Note that serialized object is deserialized as a different object.
        # Thus we can resue the same runner without state collision, without building it per thread.
        serialized_runner = dill.dumps(pass_runner)

        # TODO support for List(output_name) and List(callback)
        del run_options

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
        in_program: INPUT_TYPE,
        serialized_pass_runner: bytes = None,
    ) -> TARGET_TYPE:
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

    def passes(self) -> List[Dict[str, PASS_TYPE]]:
        """Return a list structure of the appended passes and its options.

        Returns:
            A list of pass sets, as defined in ``append()``.
        """
        ret = []
        for pass_set in self._pass_sets:
            item = {"passes": pass_set["passes"]}
            if pass_set["flow_controllers"]:
                item["flow_controllers"] = set(pass_set["flow_controllers"].keys())
            else:
                item["flow_controllers"] = {}
            ret.append(item)
        return ret

    def to_flow_controller(self) -> FlowController:
        """Linearize this manager into a single :class:`.FlowController`, so that it can be nested
        inside another :class:`.PassManager`."""
        return FlowController.controller_factory(
            [
                FlowController.controller_factory(
                    pass_set["passes"], None, **pass_set["flow_controllers"]
                )
                for pass_set in self._pass_sets
            ],
            None,
        )
