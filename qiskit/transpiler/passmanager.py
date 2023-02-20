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

import io
import re
from typing import Union, List, Tuple, Callable, Dict, Any, Optional, Iterator, Iterable

from qiskit.circuit import QuantumCircuit
from qiskit.passmanager.flow_controller import PassSequence, FlowController, FlowControllerLinear
from qiskit.passmanager.base_passmanager import BasePassManager

from .basepasses import BasePass
from .exceptions import TranspilerError
from .runningpassmanager import RunningPassManager


class PassManager(BasePassManager, passmanager_error=TranspilerError):
    """Manager for a set of Passes and their scheduling during transpilation."""

    # pylint: disable=arguments-differ
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
            circuits = [circuits]

        return super().run(
            in_programs=circuits,
            output_name=output_name,
            callback=callback,
        )

    def _create_running_passmanager(self) -> RunningPassManager:
        running_passmanager = RunningPassManager(self.max_iteration)
        for flow_controller in self._flow_controllers:
            running_passmanager.append(flow_controller)
        return running_passmanager

    def append(
        self,
        passes: PassSequence,
        max_iteration: int = None,
        **flow_controller_conditions: Callable,
    ) -> None:
        if max_iteration:
            # TODO remove this argument from append
            self.max_iteration = max_iteration
        super().append(passes, **flow_controller_conditions)

    def replace(
        self,
        index: int,
        passes: PassSequence,
        max_iteration: int = None,
        **flow_controller_conditions: Any,
    ) -> None:
        if max_iteration:
            # TODO remove this argument from append
            self.max_iteration = max_iteration
        super().replace(index, passes, **flow_controller_conditions)

    def passes(self) -> List[Dict[str, BasePass]]:
        """Return a list structure of the appended passes and its options.

        Returns:
            A list of pass sets, as defined in ``append()``.
        """
        ret = []
        for controller in self._flow_controllers:
            passes = controller.passes
            while hasattr(passes, "passes"):
                passes = passes.passes
            if isinstance(passes, BasePass):
                passes = [passes]
            if isinstance(controller, FlowControllerLinear):
                condition = {}
            else:
                for name, cls_controller in FlowController.registered_controllers.items():
                    if isinstance(controller, cls_controller):
                        condition = {name}
                        break
                else:
                    condition = {controller.__class__.__name__}
            ret.append({"passes": passes, "flow_controllers": condition})
        return ret


class StagedPassManager(PassManager):
    """A Pass manager pipeline built up of individual stages

    This class enables building a compilation pipeline out of fixed stages.
    Each ``StagedPassManager`` defines a list of stages which are executed in
    a fixed order, and each stage is defined as a standalone :class:`~.PassManager`
    instance. There are also ``pre_`` and ``post_`` stages for each defined stage.
    This enables easily composing and replacing different stages and also adding
    hook points to enable programmatic modifications to a pipeline. When using a staged
    pass manager you are not able to modify the individual passes and are only able
    to modify stages.

    By default instances of ``StagedPassManager`` define a typical full compilation
    pipeline from an abstract virtual circuit to one that is optimized and
    capable of running on the specified backend. The default pre-defined stages are:

    #. ``init`` - any initial passes that are run before we start embedding the circuit to the backend
    #. ``layout`` - This stage runs layout and maps the virtual qubits in the
       circuit to the physical qubits on a backend
    #. ``routing`` - This stage runs after a layout has been run and will insert any
       necessary gates to move the qubit states around until it can be run on
       backend's coupling map.
    #. ``translation`` - Perform the basis gate translation, in other words translate the gates
       in the circuit to the target backend's basis set
    #. ``optimization`` - The main optimization loop, this will typically run in a loop trying to
       optimize the circuit until a condition (such as fixed depth) is reached.
    #. ``scheduling`` - Any hardware aware scheduling passes

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

    def __init__(self, stages: Optional[Iterable[str]] = None, **kwargs) -> None:
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

    def _validate_init_kwargs(self, kwargs: Dict[str, Any]) -> None:
        expanded_stages = set(self.expanded_stages)
        for stage in kwargs.keys():
            if stage not in expanded_stages:
                raise AttributeError(f"{stage} is not a valid stage.")

    @property
    def stages(self) -> Tuple[str, ...]:
        """Pass manager stages"""
        return self._stages  # pylint: disable=no-member

    @property
    def expanded_stages(self) -> Tuple[str, ...]:
        """Expanded Pass manager stages including ``pre_`` and ``post_`` phases."""
        return self._expanded_stages  # pylint: disable=no-member

    def _generate_expanded_stages(self) -> Iterator[str]:
        for stage in self.stages:
            yield "pre_" + stage
            yield stage
            yield "post_" + stage

    def _update_passmanager(self) -> None:
        self._flow_controllers = []
        for stage in self.expanded_stages:
            pm = getattr(self, stage, None)
            if pm is not None:
                self._flow_controllers.extend(pm._flow_controllers)

    def __setattr__(self, attr, value):
        if value == self and attr in self.expanded_stages:
            raise TranspilerError("Recursive definition of StagedPassManager disallowed.")
        super().__setattr__(attr, value)
        if attr in self.expanded_stages:
            self._update_passmanager()

    def append(
        self,
        passes: Union[BasePass, List[BasePass]],
        max_iteration: int = None,
        **flow_controller_conditions: Any,
    ) -> None:
        raise NotImplementedError

    def replace(
        self,
        index: int,
        passes: Union[BasePass, List[BasePass]],
        max_iteration: int = None,
        **flow_controller_conditions: Any,
    ) -> None:
        raise NotImplementedError

    # Raise NotImplemntedError on individual pass manipulation
    def remove(self, index: int) -> None:
        raise NotImplementedError

    def __getitem__(self, index):
        self._update_passmanager()

        # Do not inherit from the PassManager.
        # It returns instance of self.__class__ which is StagetPassManager.
        new_passmanager = PassManager()
        new_passmanager._flow_controllers = [self._flow_controllers[index]]

        return new_passmanager

    def __len__(self):
        self._update_passmanager()
        return super().__len__()

    def __setitem__(self, index, item):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def _create_running_passmanager(self) -> RunningPassManager:
        self._update_passmanager()
        return super()._create_running_passmanager()

    def passes(self) -> List[Dict[str, BasePass]]:
        self._update_passmanager()
        return super().passes()

    def run(
        self,
        circuits: Union[QuantumCircuit, List[QuantumCircuit]],
        output_name: str = None,
        callback: Callable = None,
    ) -> Union[QuantumCircuit, List[QuantumCircuit]]:
        self._update_passmanager()
        return super().run(circuits, output_name, callback)

    def draw(self, filename=None, style=None, raw=False):
        """Draw the staged pass manager."""
        from qiskit.visualization import staged_pass_manager_drawer

        return staged_pass_manager_drawer(self, filename=filename, style=style, raw=raw)
