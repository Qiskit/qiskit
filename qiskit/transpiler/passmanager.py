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
from __future__ import annotations

import inspect
import io
import re
import warnings
from collections.abc import Iterator, Iterable, Callable
from functools import wraps
from typing import Union, List, Any

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager.passmanager import BasePassManager
from qiskit.passmanager.base_tasks import Task, BaseController
from qiskit.passmanager.flow_controllers import FlowController
from qiskit.passmanager.exceptions import PassManagerError
from qiskit.utils.deprecation import deprecate_arg
from .basepasses import BasePass
from .exceptions import TranspilerError
from .layout import TranspileLayout
from .runningpassmanager import RunningPassManager

_CircuitsT = Union[List[QuantumCircuit], QuantumCircuit]


class PassManager(BasePassManager):
    """Manager for a set of Passes and their scheduling during transpilation."""

    def __init__(
        self,
        passes: Task | list[Task] = (),
        max_iteration: int = 1000,
    ):
        """Initialize an empty pass manager object.

        Args:
            passes: A pass set to be added to the pass manager schedule.
            max_iteration: The maximum number of iterations the schedule will be looped if the
                condition is not met.
        """
        # For backward compatibility.
        self._pass_sets = []

        super().__init__(
            tasks=passes,
            max_iteration=max_iteration,
        )

    def _passmanager_frontend(
        self,
        input_program: QuantumCircuit,
        **kwargs,
    ) -> DAGCircuit:
        return circuit_to_dag(input_program, copy_operations=True)

    def _passmanager_backend(
        self,
        passmanager_ir: DAGCircuit,
        in_program: QuantumCircuit,
        **kwargs,
    ) -> QuantumCircuit:
        out_program = dag_to_circuit(passmanager_ir, copy_operations=False)

        out_name = kwargs.get("output_name", None)
        if out_name is not None:
            out_program.name = out_name

        if self.property_set["layout"] is not None:
            out_program._layout = TranspileLayout(
                initial_layout=self.property_set["layout"],
                input_qubit_mapping=self.property_set["original_qubit_indices"],
                final_layout=self.property_set["final_layout"],
                _input_qubit_count=len(in_program.qubits),
                _output_qubit_list=out_program.qubits,
            )
        out_program._clbit_write_latency = self.property_set["clbit_write_latency"]
        out_program._conditional_latency = self.property_set["conditional_latency"]

        if self.property_set["node_start_time"]:
            # This is dictionary keyed on the DAGOpNode, which is invalidated once
            # dag is converted into circuit. So this schedule information is
            # also converted into list with the same ordering with circuit.data.
            topological_start_times = []
            start_times = self.property_set["node_start_time"]
            for dag_node in passmanager_ir.topological_op_nodes():
                topological_start_times.append(start_times[dag_node])
            out_program._op_start_times = topological_start_times

        return out_program

    @deprecate_arg(
        name="max_iteration",
        since="0.25",
        additional_msg="'max_iteration' can be set in the constructor.",
        pending=True,
        package_name="qiskit-terra",
    )
    def append(
        self,
        passes: Task | list[Task],
        max_iteration: int = None,
        **flow_controller_conditions: Any,
    ) -> None:
        """Append a Pass Set to the schedule of passes.

        Args:
            passes: A set of passes (a pass set) to be added to schedule. A pass set is a list of
                passes that are controlled by the same flow controller. If a single pass is
                provided, the pass set will only have that pass a single element.
                It is also possible to append a :class:`.BaseFlowController` instance and
                the rest of the parameter will be ignored.
            max_iteration: max number of iterations of passes.
            flow_controller_conditions: Dictionary of control flow plugins.
                Following built-in controllers are available by default:

                * do_while: The passes repeat until the callable returns False.  Corresponds to
                  :class:`.DoWhileController`.
                * condition: The passes run only if the callable returns True.  Corresponds to
                  :class:`.ConditionalController`.

                In general, you have more control simply by creating the controller you want and
                passing it to :meth:`append`.

        Raises:
            TranspilerError: if a pass in passes is not a proper pass.
        """
        if max_iteration:
            self.max_iteration = max_iteration

        # Backward compatibility as of Terra 0.25
        if isinstance(passes, Task):
            passes = [passes]
        self._pass_sets.append(
            {
                "passes": passes,
                "flow_controllers": flow_controller_conditions,
            }
        )
        if flow_controller_conditions:
            passes = _legacy_build_flow_controller(
                passes,
                options={"max_iteration": self.max_iteration},
                **flow_controller_conditions,
            )

        super().append(passes)

    @deprecate_arg(
        name="max_iteration",
        since="0.25",
        additional_msg="'max_iteration' can be set in the constructor.",
        pending=True,
        package_name="qiskit-terra",
    )
    def replace(
        self,
        index: int,
        passes: Task | list[Task],
        max_iteration: int = None,
        **flow_controller_conditions: Any,
    ) -> None:
        """Replace a particular pass in the scheduler.

        Args:
            index: Pass index to replace, based on the position in passes().
            passes: A pass set to be added to the pass manager schedule.
            max_iteration: max number of iterations of passes.
            flow_controller_conditions: Dictionary of control flow plugins.
                See :meth:`qiskit.transpiler.PassManager.append` for details.
        """
        if max_iteration:
            self.max_iteration = max_iteration

        # Backward compatibility as of Terra 0.25
        if isinstance(passes, Task):
            passes = [passes]
        try:
            self._pass_sets[index] = {
                "passes": passes,
                "flow_controllers": flow_controller_conditions,
            }
        except IndexError as ex:
            raise PassManagerError(f"Index to replace {index} does not exists") from ex
        if flow_controller_conditions:
            passes = _legacy_build_flow_controller(
                passes,
                options={"max_iteration": self.max_iteration},
                **flow_controller_conditions,
            )

        super().replace(index, passes)

    def remove(self, index: int) -> None:
        super().remove(index)

        # Backward compatibility as of Terra 0.25
        del self._pass_sets[index]

    def __getitem__(self, index):
        new_passmanager = super().__getitem__(index)

        # Backward compatibility as of Terra 0.25
        _pass_sets = self._pass_sets[index]
        if isinstance(_pass_sets, dict):
            _pass_sets = [_pass_sets]
        new_passmanager._pass_sets = _pass_sets
        return new_passmanager

    def __add__(self, other):
        new_passmanager = super().__add__(other)

        # Backward compatibility as of Terra 0.25
        if isinstance(other, self.__class__):
            new_passmanager._pass_sets = self._pass_sets
            new_passmanager._pass_sets += other._pass_sets

        # When other is not identical type, _pass_sets is also evaluated by self.append.
        return new_passmanager

    def to_flow_controller(self) -> RunningPassManager:
        # For backward compatibility.
        # This method will be resolved to the base class and return FlowControllerLinear
        flatten_tasks = list(self._flatten_tasks(self._tasks))
        return RunningPassManager(flatten_tasks)

    # pylint: disable=arguments-differ
    def run(
        self,
        circuits: _CircuitsT,
        output_name: str | None = None,
        callback: Callable = None,
    ) -> _CircuitsT:
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

                .. note::

                    Beware that the keyword arguments here are different to those used by the
                    generic :class:`.BasePassManager`.  This pass manager will translate those
                    arguments into the form described above.

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
        if callback is not None:
            callback = _legacy_style_callback(callback)

        return super().run(
            in_programs=circuits,
            callback=callback,
            output_name=output_name,
        )

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

    def passes(self) -> list[dict[str, BasePass]]:
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


class StagedPassManager(PassManager):
    """A pass manager pipeline built from individual stages.

    This class enables building a compilation pipeline out of fixed stages.
    Each ``StagedPassManager`` defines a list of stages which are executed in
    a fixed order, and each stage is defined as a standalone :class:`~.PassManager`
    instance. There are also ``pre_`` and ``post_`` stages for each defined stage.
    This enables easily composing and replacing different stages and also adding
    hook points to enable programmatic modifications to a pipeline. When using a staged
    pass manager you are not able to modify the individual passes and are only able
    to modify stages.

    By default, instances of ``StagedPassManager`` define a typical full compilation
    pipeline from an abstract virtual circuit to one that is optimized and
    capable of running on the specified backend. The default pre-defined stages are:

    #. ``init`` - Initial passes to run before embedding the circuit to the backend.
    #. ``layout`` - Maps the virtual qubits in the circuit to the physical qubits on
       the backend.
    #. ``routing`` - Inserts gates as needed to move the qubit states around until
       the circuit can be run with the chosen layout on the backend's coupling map.
    #. ``translation`` - Translates the gates in the circuit to the target backend's
       basis gate set.
    #. ``optimization`` - Optimizes the circuit to reduce the cost of executing it.
       These passes will typically run in a loop until a convergence criteria is met.
       For example, the convergence criteria might be that the circuit depth does not
       decrease in successive iterations.
    #. ``scheduling`` - Hardware-aware passes that schedule the operations in the
       circuit.

    .. note::

        For backwards compatibility the relative positioning of these default
        stages will remain stable moving forward. However, new stages may be
        added to the default stage list in between current stages. For example,
        in a future release a new phase, something like ``logical_optimization``, could be added
        immediately after the existing ``init`` stage in the default stage list.
        This would preserve compatibility for pre-existing ``StagedPassManager``
        users as the relative positions of the stage are preserved so the behavior
        will not change between releases.

    These stages will be executed in order and any stage set to ``None`` will be skipped.
    If a stage is provided multiple times (i.e. at diferent relative positions), the
    associated passes, including pre and post, will run once per declaration.
    If a :class:`~qiskit.transpiler.PassManager` input is being used for more than 1 stage here
    (for example in the case of a :class:`~.Pass` that covers both Layout and Routing) you will
    want to set that to the earliest stage in sequence that it covers.
    """

    invalid_stage_regex = re.compile(
        r"\s|\+|\-|\*|\/|\\|\%|\<|\>|\@|\!|\~|\^|\&|\:|\[|\]|\{|\}|\(|\)"
    )

    def __init__(self, stages: Iterable[str] | None = None, **kwargs) -> None:
        """Initialize a new StagedPassManager object

        Args:
            stages (Iterable[str]): An optional list of stages to use for this
                instance. If this is not specified the default stages list
                ``['init', 'layout', 'routing', 'translation', 'optimization', 'scheduling']`` is
                used. After instantiation, the final list will be immutable and stored as tuple.
                If a stage is provided multiple times (i.e. at diferent relative positions), the
                associated passes, including pre and post, will run once per declaration.
            kwargs: The initial :class:`~.PassManager` values for any stages
                defined in ``stages``. If a argument is not defined the
                stages will default to ``None`` indicating an empty/undefined
                stage.

        Raises:
            AttributeError: If a stage in the input keyword arguments is not defined.
            ValueError: If an invalid stage name is specified.
        """
        stages = stages or [
            "init",
            "layout",
            "routing",
            "translation",
            "optimization",
            "scheduling",
        ]
        self._validate_stages(stages)
        # Set through parent class since `__setattr__` requieres `expanded_stages` to be defined
        super().__setattr__("_stages", tuple(stages))
        super().__setattr__("_expanded_stages", tuple(self._generate_expanded_stages()))
        super().__init__()
        self._validate_init_kwargs(kwargs)
        for stage in set(self.expanded_stages):
            pm = kwargs.get(stage, None)
            setattr(self, stage, pm)

    def _validate_stages(self, stages: Iterable[str]) -> None:
        invalid_stages = [
            stage for stage in stages if self.invalid_stage_regex.search(stage) is not None
        ]
        if invalid_stages:
            with io.StringIO() as msg:
                msg.write(f"The following stage names are not valid: {invalid_stages[0]}")
                for invalid_stage in invalid_stages[1:]:
                    msg.write(f", {invalid_stage}")
                raise ValueError(msg.getvalue())

    def _validate_init_kwargs(self, kwargs: dict[str, Any]) -> None:
        expanded_stages = set(self.expanded_stages)
        for stage in kwargs.keys():
            if stage not in expanded_stages:
                raise AttributeError(f"{stage} is not a valid stage.")

    @property
    def stages(self) -> tuple[str, ...]:
        """Pass manager stages"""
        return self._stages  # pylint: disable=no-member

    @property
    def expanded_stages(self) -> tuple[str, ...]:
        """Expanded Pass manager stages including ``pre_`` and ``post_`` phases."""
        return self._expanded_stages  # pylint: disable=no-member

    def _generate_expanded_stages(self) -> Iterator[str]:
        for stage in self.stages:
            yield "pre_" + stage
            yield stage
            yield "post_" + stage

    def _update_passmanager(self) -> None:
        self._tasks = []
        self._pass_sets = []
        for stage in self.expanded_stages:
            pm = getattr(self, stage, None)
            if pm is not None:
                self._tasks += pm._tasks
                self._pass_sets.extend(pm._pass_sets)

    def __setattr__(self, attr, value):
        if value == self and attr in self.expanded_stages:
            raise TranspilerError("Recursive definition of StagedPassManager disallowed.")
        super().__setattr__(attr, value)
        if attr in self.expanded_stages:
            self._update_passmanager()

    def append(
        self,
        passes: Task | list[Task],
        max_iteration: int = None,
        **flow_controller_conditions: Any,
    ) -> None:
        raise NotImplementedError

    def replace(
        self,
        index: int,
        passes: BasePass | list[BasePass],
        max_iteration: int = None,
        **flow_controller_conditions: Any,
    ) -> None:
        raise NotImplementedError

    # Raise NotImplemntedError on individual pass manipulation
    def remove(self, index: int) -> None:
        raise NotImplementedError

    def __getitem__(self, index):
        self._update_passmanager()

        # Do not inherit from the PassManager, i.e. super()
        # It returns instance of self.__class__ which is StagedPassManager.
        new_passmanager = PassManager(max_iteration=self.max_iteration)
        new_passmanager._tasks = self._tasks[index]
        _pass_sets = self._pass_sets[index]
        if isinstance(_pass_sets, dict):
            _pass_sets = [_pass_sets]
        new_passmanager._pass_sets = _pass_sets
        return new_passmanager

    def __len__(self):
        self._update_passmanager()
        return super().__len__()

    def __setitem__(self, index, item):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def passes(self) -> list[dict[str, BasePass]]:
        self._update_passmanager()
        return super().passes()

    def run(
        self,
        circuits: _CircuitsT,
        output_name: str | None = None,
        callback: Callable | None = None,
    ) -> _CircuitsT:
        self._update_passmanager()
        return super().run(circuits, output_name, callback)

    def draw(self, filename=None, style=None, raw=False):
        """Draw the staged pass manager."""
        from qiskit.visualization import staged_pass_manager_drawer

        return staged_pass_manager_drawer(self, filename=filename, style=style, raw=raw)


# A temporary error handling with slight overhead at class loading.
# This method wraps all class methods to replace PassManagerError with TranspilerError.
# The pass flow controller mechanics raises PassManagerError, as it has been moved to base class.
# PassManagerError is not caught by TranspilerError due to the hierarchy.


def _replace_error(meth):
    @wraps(meth)
    def wrapper(*meth_args, **meth_kwargs):
        try:
            return meth(*meth_args, **meth_kwargs)
        except PassManagerError as ex:
            raise TranspilerError(ex.message) from ex

    return wrapper


for _name, _method in inspect.getmembers(PassManager, predicate=inspect.isfunction):
    if _name.startswith("_"):
        # Ignore protected and private.
        # User usually doesn't directly execute and catch error from these methods.
        continue
    _wrapped = _replace_error(_method)
    setattr(PassManager, _name, _wrapped)


def _legacy_style_callback(callback: Callable):
    def _wrapped_callable(task, passmanager_ir, property_set, running_time, count):
        callback(
            pass_=task,
            dag=passmanager_ir,
            time=running_time,
            property_set=property_set,
            count=count,
        )

    return _wrapped_callable


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
        stacklevel=3,
    )
    if isinstance(tasks, Task):
        tasks = [tasks]
    if any(not isinstance(t, Task) for t in tasks):
        raise TypeError("Added tasks are not all valid pass manager task types.")
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
