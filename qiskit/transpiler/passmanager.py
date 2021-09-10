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

from typing import Union, List, Callable, Dict, Any

import dill

from qiskit.visualization import pass_manager_drawer
from qiskit.tools.parallel import parallel_map
from qiskit.circuit import QuantumCircuit
from .basepasses import BasePass
from .exceptions import TranspilerError
from .runningpassmanager import RunningPassManager, FlowController


class PassManager:
    """Manager for a set of Passes and their scheduling during transpilation."""

    def __init__(self, passes: Union[BasePass, List[BasePass]] = None, max_iteration: int = 1000):
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
        passes: Union[BasePass, List[BasePass]],
        max_iteration: int = None,
        **flow_controller_conditions: Any,
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
        self._pass_sets.append({"passes": passes, "flow_controllers": flow_controller_conditions})

    def replace(
        self,
        index: int,
        passes: Union[BasePass, List[BasePass]],
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
            TranspilerError: if a pass in passes is not a proper pass or index not found.

        See Also:
            ``RunningPassManager.add_flow_controller()`` for more information about the control
            flow plugins.
        """
        if max_iteration:
            # TODO remove this argument from append
            self.max_iteration = max_iteration

        passes = PassManager._normalize_passes(passes)

        try:
            self._pass_sets[index] = {
                "passes": passes,
                "flow_controllers": flow_controller_conditions,
            }
        except IndexError as ex:
            raise TranspilerError(f"Index to replace {index} does not exists") from ex

    def remove(self, index: int) -> None:
        """Removes a particular pass in the scheduler.

        Args:
            index: Pass index to replace, based on the position in passes().

        Raises:
            TranspilerError: if the index is not found.
        """
        try:
            del self._pass_sets[index]
        except IndexError as ex:
            raise TranspilerError(f"Index to replace {index} does not exists") from ex

    def __setitem__(self, index, item):
        self.replace(index, item)

    def __len__(self):
        return len(self._pass_sets)

    def __getitem__(self, index):
        new_passmanager = PassManager(max_iteration=self.max_iteration)
        _pass_sets = self._pass_sets[index]
        if isinstance(_pass_sets, dict):
            _pass_sets = [_pass_sets]
        new_passmanager._pass_sets = _pass_sets
        return new_passmanager

    def __add__(self, other):
        if isinstance(other, PassManager):
            new_passmanager = PassManager(max_iteration=self.max_iteration)
            new_passmanager._pass_sets = self._pass_sets + other._pass_sets
            return new_passmanager
        else:
            try:
                new_passmanager = PassManager(max_iteration=self.max_iteration)
                new_passmanager._pass_sets += self._pass_sets
                new_passmanager.append(other)
                return new_passmanager
            except TranspilerError as ex:
                raise TypeError(
                    f"unsupported operand type + for {self.__class__} and {other.__class__}"
                ) from ex

    @staticmethod
    def _normalize_passes(
        passes: Union[BasePass, List[BasePass], FlowController]
    ) -> List[BasePass]:
        if isinstance(passes, FlowController):
            return passes
        if isinstance(passes, BasePass):
            passes = [passes]
        for pass_ in passes:
            if not isinstance(pass_, BasePass):
                raise TranspilerError("%s is not a pass instance" % pass_.__class__)
        return passes

    def run(
        self,
        circuits: Union[QuantumCircuit, List[QuantumCircuit]],
        output_name: str = None,
        callback: Callable = None,
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
            running_passmanager.append(pass_set["passes"], **pass_set["flow_controllers"])
        return running_passmanager

    @staticmethod
    def _in_parallel(circuit, pm_dill=None) -> QuantumCircuit:
        """Task used by the parallel map tools from ``_run_several_circuits``."""
        running_passmanager = dill.loads(pm_dill)._create_running_passmanager()
        result = running_passmanager.run(circuit)
        return result

    def _run_several_circuits(
        self, circuits: List[QuantumCircuit], output_name: str = None, callback: Callable = None
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

        return parallel_map(
            PassManager._in_parallel, circuits, task_kwargs={"pm_dill": dill.dumps(self)}
        )

    def _run_single_circuit(
        self, circuit: QuantumCircuit, output_name: str = None, callback: Callable = None
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
            item = {"passes": pass_set["passes"]}
            if pass_set["flow_controllers"]:
                item["flow_controllers"] = set(pass_set["flow_controllers"].keys())
            else:
                item["flow_controllers"] = {}
            ret.append(item)
        return ret


class FullPassManager(PassManager):
    """A full Pass manager pipeline for a backend

    Instances of FullPassManager define a full compilation pipeline from a abstract virtual
    circuit to one that is optimized and capable of running on the specified backend. It is
    built using predefined stages:

    1. Init - any initial passes that are run before we start embedding the circuit to the backend
    2. Layout - This stage runs layout and maps the virtual qubits in the
        circuit to the physical qubits on a backend
    3. Routing - This stage runs after a layout has been run and will insert any
        necessary gates to move the qubit states around until it can be run on
        backend's compuling map.
    4. Translation - Perform the basis gate translation, in other words translate the gates
        in the circuit to the target backend's basis set
    5. Pre-Optimization - Any passes to run before the main optimization loop
    6. Optimization - The main optimization loop, this will typically run in a loop trying to optimize
        the circuit until a condtion (such as fixed depth) is reached.
    7. Post-Optimization - Any passes to run after the main optimization loop
    8. Scheduling - Any hardware aware scheduling passes

    These stages will be executed in order and any stage set to ``None`` will be skipped. If
    a :class:`~qiskit.transpiler.PassManager` input is being used for more than 1 stage here
    (for example in the case of a Pass that covers both Layout and Routing) you will want to set
    that to the earliest stage in sequence that it covers.
    """

    phases = [
        "init",
        "layout",
        "routing",
        "translation",
        "pre_optimization",
        "optimization",
        "post_optimization",
        "scheduling",
    ]

    def __init__(
        self,
        init=None,
        layout=None,
        routing=None,
        translation=None,
        pre_optimization=None,
        optimization=None,
        post_optimization=None,
        scheduling=None,
    ):
        """Initialize a new FullPassManager object

        Args:
            init (PassManager): A passmanager to run for the initial stage of the
                compilation.
            layout (PassManager): A passmanager to run for the layout stage of the
                compilation.
            routing (PassManager): A pass manager to run for the routing stage
                of the compilation
            translation (PassManager): A pass manager to run for the translation
                stage of the compilation
            pre_opt (PassManager): A pass manager to run before the optimization
                loop
            optimization (PassManager): A pass manager to run for the
                optimization loop stage
            post_opt (PassManager): A pass manager to run after the optimization
                loop
            scheduling (PassManager): A pass manager to run any scheduling passes
        """
        super().__init__()
        self._init = init
        self._layout = layout
        self._routing = routing
        self._translation = translation
        self._pre_optimization = pre_optimization
        self._optimization = optimization
        self._post_optimization = post_optimization
        self._scheduling = scheduling
        self._update_passmanager()

    def _update_passmanager(self):
        self._pass_sets = []
        if self._init:
            self._pass_sets.extend(self._init._pass_sets)
        if self._layout:
            self._pass_sets.extend(self._layout._pass_sets)
        if self._routing:
            self._pass_sets.extend(self._routing._pass_sets)
        if self._translation:
            self._pass_sets.extend(self._translation._pass_sets)
        if self._pre_optimization:
            self._pass_sets.extend(self._pre_optimization._pass_sets)
        if self._optimization:
            self._pass_sets.extend(self._optimization._pass_sets)
        if self._post_optimization:
            self._pass_sets.extend(self._post_optimization._pass_sets)
        if self._scheduling:
            self._pass_sets.extend(self._scheduling._pass_sets)

    @property
    def init(self):
        """Get the :class:`~qiskit.transpiler.PassManager` for the init stage."""
        return self._init

    @init.setter
    def init(self, value):
        """Set the :class:`~qiskit.transpiler.PassManager` for the init stage."""
        self._init = value
        self._update_passmanager()

    @property
    def layout(self):
        """Get the :class:`~qiskit.transpiler.PassManager` for the layout stage."""
        return self._layout

    @layout.setter
    def layout(self, value):
        """Set the :class:`~qiskit.transpiler.PassManager` for the layout stage."""
        self._layout = value
        self._update_passmanager()

    @property
    def routing(self):
        """Get the :class:`~qiskit.transpiler.PassManager` for the routing stage."""
        return self._routing

    @routing.setter
    def routing(self, value):
        """Set the :class:`~qiskit.transpiler.PassManager` for the routing stage."""
        self._routing = value
        self._update_passmanager()

    @property
    def pre_optimization(self):
        """Get the :class:`~qiskit.transpiler.PassManager` for the pre_optimization stage."""
        return self._pre_optimization

    @pre_optimization.setter
    def pre_optimization(self, value):
        """Set the :class:`~qiskit.transpiler.PassManager` for the pre_optimization stage."""
        self._pre_optimization = value
        self._update_passmanager()

    @property
    def optimization(self):
        """Get the :class:`~qiskit.transpiler.PassManager` for the optimization stage."""
        return self._optimization

    @optimization.setter
    def optimization(self, value):
        """Set the :class:`~qiskit.transpiler.PassManager` for the optimization stage."""
        self._optimization = value
        self._update_passmanager()

    @property
    def post_optimization(self):
        """Get the :class:`~qiskit.transpiler.PassManager` for the post_optimization stage."""
        return self._post_optimization

    @post_optimization.setter
    def post_optimization(self, value):
        """Set the :class:`~qiskit.transpiler.PassManager` for the post_optimization stage."""
        self._post_optimization = value
        self._update_passmanager()

    @property
    def scheduling(self):
        """Get the :class:`~qiskit.transpiler.PassManager` for the scheduling stage."""
        return self._post_optimization

    @scheduling.setter
    def scheduling(self, value):
        """Set the :class:`~qiskit.transpiler.PassManager` for the scheduling stage."""
        self._post_optimization = value
        self._update_passmanager()
