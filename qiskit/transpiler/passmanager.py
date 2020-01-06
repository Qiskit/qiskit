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

"""PassManager class for the transpiler."""

import dill

from qiskit.visualization import pass_manager_drawer
from qiskit.tools.parallel import parallel_map
from qiskit.circuit import QuantumCircuit
from .basepasses import BasePass
from .exceptions import TranspilerError
from .runningpassmanager import RunningPassManager


class PassManager:
    """A PassManager schedules the passes"""

    def __init__(self, passes=None, max_iteration=1000, callback=None):
        """Initialize an empty PassManager object (with no passes scheduled).

        Args:
            passes (list[BasePass] or BasePass): A pass set (as defined in ``append()``)
                to be added to the pass manager schedule. The default is None.
            max_iteration (int): The schedule looping iterates until the condition is met or until
                max_iteration is reached.
            callback (func): A callback function that will be called after each
                pass execution. The function will be called with 5 keyword
                arguments::
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

                    PassManager(callback=callback_func)

        """
        self._pass_sets = []
        if passes is not None:
            self.append(passes)
        self.max_iteration = max_iteration
        self.callback = callback
        self.property_set = None

    def append(self, passes, max_iteration=None, **flow_controller_conditions):
        """Append a Pass Set to the schedule of passes.

        Args:
            passes (list[BasePass] or BasePass): A set of passes (a pass set) to be added
               to schedule. A pass set is a list of passes that are controlled by the same
               flow controller. If a single pass is provided, the pass set will only have that
               pass a single element.
            max_iteration (int): max number of iterations of passes. Default: 1000
            flow_controller_conditions (kwargs): See add_flow_controller(): Dictionary of
            control flow plugins. Default:

                * do_while (callable property_set -> boolean): The passes repeat until the
                  callable returns False.
                  Default: `lambda x: False # i.e. passes run once`

                * condition (callable property_set -> boolean): The passes run only if the
                  callable returns True.
                  Default: `lambda x: True # i.e. passes run`

        Raises:
            TranspilerError: if a pass in passes is not a proper pass.
        """
        if max_iteration:
            # TODO remove this argument from append
            self.max_iteration = max_iteration

        passes = PassManager._normalize_passes(passes)
        self._pass_sets.append({'passes': passes, 'flow_controllers': flow_controller_conditions})

    def replace(self, index, passes, max_iteration=None, **flow_controller_conditions):
        """Replace a particular pass in the scheduler

        Args:
            index (int): Pass index to replace, based on the position in passes().
            passes (list[BasePass] or BasePass): A pass set (as defined in ``append()``)
                   to be added to the pass manager schedule
            max_iteration (int): max number of iterations of passes. Default: 1000
            flow_controller_conditions (kwargs): See add_flow_controller(): Dictionary of
            control flow plugins. Default:

                * do_while (callable property_set -> boolean): The passes repeat until the
                  callable returns False.
                  Default: `lambda x: False # i.e. passes run once`

                * condition (callable property_set -> boolean): The passes run only if the
                  callable returns True.
                  Default: `lambda x: True # i.e. passes run`
        Raises:
            TranspilerError: if a pass in passes is not a proper pass.
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
        max_iteration = self.max_iteration
        call_back = self.callback
        new_passmanager = PassManager(max_iteration=max_iteration, callback=call_back)
        _pass_sets = self._pass_sets[index]
        if isinstance(_pass_sets, dict):
            _pass_sets = [_pass_sets]
        new_passmanager._pass_sets = _pass_sets
        return new_passmanager

    def __add__(self, other):
        if isinstance(other, PassManager):
            max_iteration = self.max_iteration
            call_back = self.callback
            new_passmanager = PassManager(max_iteration=max_iteration, callback=call_back)
            new_passmanager._pass_sets = self._pass_sets + other._pass_sets
            return new_passmanager
        else:
            try:
                max_iteration = self.max_iteration
                call_back = self.callback
                new_passmanager = PassManager(max_iteration=max_iteration, callback=call_back)
                new_passmanager._pass_sets += self._pass_sets
                new_passmanager.append(other)
                return new_passmanager
            except TranspilerError:
                raise TypeError('unsupported operand type + for %s and %s' % (self.__class__,
                                                                              other.__class__))

    @staticmethod
    def _normalize_passes(passes):
        if isinstance(passes, BasePass):
            passes = [passes]

        for pass_ in passes:
            if not isinstance(pass_, BasePass):
                raise TranspilerError('%s is not a pass instance' % pass_.__class__)
        return passes

    def run(self, circuits):
        """Run all the passes on circuit or circuits

        Args:
            circuits (QuantumCircuit or list[QuantumCircuit]): circuit(s) to
            transform via all the registered passes.

        Returns:
            QuantumCircuit or list[QuantumCircuit]: Transformed circuit(s).
        """
        if isinstance(circuits, QuantumCircuit):
            return self._run_single_circuit(circuits)
        else:
            return self._run_several_circuits(circuits)

    def _create_running_passmanager(self):
        running_passmanager = RunningPassManager(self.max_iteration, self.callback)
        for pass_set in self._pass_sets:
            running_passmanager.append(pass_set['passes'], **pass_set['flow_controllers'])
        return running_passmanager

    @staticmethod
    def _in_parallel(circuit, pm_dill=None):
        """ Used by _run_several_circuits. """
        running_passmanager = dill.loads(pm_dill)._create_running_passmanager()
        result = running_passmanager.run(circuit)
        return result

    def _run_several_circuits(self, circuits):
        """Run all the passes on each of the circuits in the circuits list

        Args:
            circuits (list[QuantumCircuit]): circuit to transform via all the registered passes

        Returns:
            list[QuantumCircuit]: Transformed circuits.
        """
        return parallel_map(PassManager._in_parallel, circuits,
                            task_kwargs={'pm_dill': dill.dumps(self)})

    def _run_single_circuit(self, circuit):
        """Run all the passes on a QuantumCircuit

        Args:
            circuit (QuantumCircuit): circuit to transform via all the registered passes

        Returns:
            QuantumCircuit: Transformed circuit.
        """
        running_passmanager = self._create_running_passmanager()
        result = running_passmanager.run(circuit)
        self.property_set = running_passmanager.property_set
        return result

    def draw(self, filename=None, style=None, raw=False):
        """
        Draws the pass manager.

        This function needs `pydot <https://github.com/erocarrera/pydot>`, which in turn needs
        Graphviz <https://www.graphviz.org/>` to be installed.

        Args:
            filename (str or None): file path to save image to
            style (dict or OrderedDict): keys are the pass classes and the values are
                the colors to make them. An example can be seen in the DEFAULT_STYLE. An ordered
                dict can be used to ensure a priority coloring when pass falls into multiple
                categories. Any values not included in the provided dict will be filled in from
                the default dict
            raw (Bool) : True if you want to save the raw Dot output not an image. The
                default is False.
        Returns:
            PIL.Image or None: an in-memory representation of the pass manager. Or None if
                               no image was generated or PIL is not installed.
        Raises:
            ImportError: when nxpd or pydot not installed.
        """
        return pass_manager_drawer(self, filename=filename, style=style, raw=raw)

    def passes(self):
        """Return a list structure of the appended passes and its options.

        Returns (list): A list of pass sets as defined in ``append()``.
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
