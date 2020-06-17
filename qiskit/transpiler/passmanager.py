# -*- coding: utf-8 -*-

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

import warnings
from typing import Union, List, Callable, Dict, Any

import dill

from qiskit.visualization import pass_manager_drawer
from qiskit.tools.parallel import parallel_map
from qiskit.circuit import QuantumCircuit
from .basepasses import BasePass
from .exceptions import TranspilerError
from .runningpassmanager import RunningPassManager


class PassManager:
    """Manager for a set of Passes and their scheduling during transpilation."""

    def __init__(
            self,
            passes: Union[BasePass, List[BasePass]] = None,
            max_iteration: int = 1000,
            callback: Callable = None
    ):
        """Initialize an empty `PassManager` object (with no passes scheduled).

        Args:
            passes: A pass set (as defined in :py:func:`qiskit.transpiler.PassManager.append`)
                to be added to the pass manager schedule.
            max_iteration: The maximum number of iterations the schedule will be looped if the
                condition is not met.
            callback: DEPRECATED - A callback function that will be called after each pass
                execution.

        .. deprecated:: 0.13.0
            The ``callback`` parameter is deprecated in favor of
            ``PassManager.run(..., callback=callback, ...)``.
        """
        self.callback = None

        if callback:
            warnings.warn("Setting a callback at construction time is being deprecated in favor of"
                          "PassManager.run(..., callback=callback,...)", DeprecationWarning, 2)
            self.callback = callback
        # the pass manager's schedule of passes, including any control-flow.
        # Populated via PassManager.append().

        self._pass_sets = []
        if passes is not None:
            self.append(passes)
        self.max_iteration = max_iteration
        self.property_set = None

    def append(
            self,
            passes: Union[BasePass, List[BasePass]],
            max_iteration: int = None,
            **flow_controller_conditions: Any
    ) -> None:
        """Append a Pass Set to the schedule of passes.

        Args:
            passes: A set of passes (a pass set) to be added to schedule. A pass set is a list of
                    passes that are controlled by the same flow controller. If a single pass is
                    provided, the pass set will only have that pass a single element.
            max_iteration: max number of iterations of passes.
            flow_controller_conditions: control flow plugins.

        Raises:
            TranspilerError: if a pass in passes is not a proper pass.

        See Also:
            ``RunningPassManager.add_flow_controller()`` for more information about the control
            flow plugins.
        """
        if max_iteration:
            # TODO remove this argument from append
            self.max_iteration = max_iteration

        passes = PassManager._normalize_passes(passes)
        self._pass_sets.append({'passes': passes, 'flow_controllers': flow_controller_conditions})

    def replace(
            self,
            index: int,
            passes: Union[BasePass, List[BasePass]],
            max_iteration: int = None,
            **flow_controller_conditions: Any
    ) -> None:
        """Replace a particular pass in the scheduler.

        Args:
            index: Pass index to replace, based on the position in passes().
            passes: A pass set (as defined in :py:func:`qiskit.transpiler.PassManager.append`)
                to be added to the pass manager schedule.
            max_iteration: max number of iterations of passes.
            flow_controller_conditions: control flow plugins.

        Raises:
            TranspilerError: if a pass in passes is not a proper pass.

        See Also:
            ``RunningPassManager.add_flow_controller()`` for more information about the control
            flow plugins.
        """
        if max_iteration:
            # TODO remove this argument from append
            self.max_iteration = max_iteration

        passes = PassManager._normalize_passes(passes)

        try:
            self._pass_sets[index] = {'passes': passes,
                                      'flow_controllers': flow_controller_conditions}
        except IndexError:
            raise TranspilerError('Index to replace %s does not exists' % index)

    def __setitem__(self, index, item):
        self.replace(index, item)

    def __len__(self):
        return len(self._pass_sets)

    def __getitem__(self, index):
        new_passmanager = PassManager(max_iteration=self.max_iteration, callback=self.callback)
        _pass_sets = self._pass_sets[index]
        if isinstance(_pass_sets, dict):
            _pass_sets = [_pass_sets]
        new_passmanager._pass_sets = _pass_sets
        return new_passmanager

    def __add__(self, other):
        if isinstance(other, PassManager):
            new_passmanager = PassManager(max_iteration=self.max_iteration, callback=self.callback)
            new_passmanager._pass_sets = self._pass_sets + other._pass_sets
            return new_passmanager
        else:
            try:
                new_passmanager = PassManager(max_iteration=self.max_iteration,
                                              callback=self.callback)
                new_passmanager._pass_sets += self._pass_sets
                new_passmanager.append(other)
                return new_passmanager
            except TranspilerError:
                raise TypeError('unsupported operand type + for %s and %s' % (self.__class__,
                                                                              other.__class__))

    @staticmethod
    def _normalize_passes(passes: Union[BasePass, List[BasePass]]) -> List[BasePass]:
        if isinstance(passes, BasePass):
            passes = [passes]

        for pass_ in passes:
            if not isinstance(pass_, BasePass):
                raise TranspilerError('%s is not a pass instance' % pass_.__class__)
        return passes

    def run(
            self,
            circuits: Union[QuantumCircuit, List[QuantumCircuit]],
            output_name: str = None,
            callback: Callable = None
    ) -> Union[QuantumCircuit, List[QuantumCircuit]]:
        """Run all the passes on the specified ``circuits``.

        Args:
            circuits: Circuit(s) to transform via all the registered passes.
            output_name: The output circuit name. If ``None``, it will be set to the same as the
                input circuit name.
            callback: A callback function that will be called after each pass execution. The
                function will be called with 5 keyword arguments::

                    pass_ (Pass): the pass being run
                    dag (DAGCircuit): the dag output of the pass
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

        Returns:
            The transformed circuit(s).
        """
        if isinstance(circuits, QuantumCircuit):
            return self._run_single_circuit(circuits, output_name, callback)
        elif len(circuits) == 1:
            return self._run_single_circuit(circuits[0], output_name, callback)
        else:
            return self._run_several_circuits(circuits, output_name, callback)

    def _create_running_passmanager(self) -> RunningPassManager:
        running_passmanager = RunningPassManager(self.max_iteration)
        for pass_set in self._pass_sets:
            running_passmanager.append(pass_set['passes'], **pass_set['flow_controllers'])
        return running_passmanager

    @staticmethod
    def _in_parallel(circuit, pm_dill=None) -> QuantumCircuit:
        """Task used by the parallel map tools from ``_run_several_circuits``."""
        running_passmanager = dill.loads(pm_dill)._create_running_passmanager()
        result = running_passmanager.run(circuit)
        return result

    def _run_several_circuits(
            self,
            circuits: List[QuantumCircuit],
            output_name: str = None,
            callback: Callable = None
    ) -> List[QuantumCircuit]:
        """Run all the passes on the specified ``circuits``.

        Args:
            circuits: Circuits to transform via all the registered passes.
            output_name: The output circuit name. If ``None``, it will be set to the same as the
                input circuit name.
            callback: A callback function that will be called after each pass execution.

        Returns:
            The transformed circuits.
        """
        # TODO support for List(output_name) and List(callback)
        del output_name
        del callback

        return parallel_map(PassManager._in_parallel, circuits,
                            task_kwargs={'pm_dill': dill.dumps(self)})

    def _run_single_circuit(
            self,
            circuit: QuantumCircuit,
            output_name: str = None,
            callback: Callable = None
    ) -> QuantumCircuit:
        """Run all the passes on a ``circuit``.

        Args:
            circuit: Circuit to transform via all the registered passes.
            output_name: The output circuit name. If ``None``, it will be set to the same as the
                input circuit name.
            callback: A callback function that will be called after each pass execution.

        Returns:
            The transformed circuit.
        """
        running_passmanager = self._create_running_passmanager()
        if callback is None and self.callback:  # TODO to remove with __init__(callback)
            callback = self.callback
        result = running_passmanager.run(circuit, output_name=output_name, callback=callback)
        self.property_set = running_passmanager.property_set
        return result

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
        return pass_manager_drawer(self, filename=filename, style=style, raw=raw)

    def passes(self) -> List[Dict[str, BasePass]]:
        """Return a list structure of the appended passes and its options.

        Returns:
            A list of pass sets, as defined in ``append()``.
        """
        ret = []
        for pass_set in self._pass_sets:
            item = {'passes': pass_set['passes']}
            if pass_set['flow_controllers']:
                item['flow_controllers'] = set(pass_set['flow_controllers'].keys())
            else:
                item['flow_controllers'] = {}
            ret.append(item)
        return ret
