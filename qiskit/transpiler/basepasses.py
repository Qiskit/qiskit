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

"""Base transpiler passes."""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Hashable, Iterable
from inspect import signature

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager.base_tasks import GenericPass, Task
from qiskit.passmanager.propertyset import PropertySet, PassState, RunState

from .exceptions import TranspilerError
from .fencedobjs import FencedDAGCircuit
from .layout import TranspileLayout


class MetaPass(type):
    """Metaclass for transpiler passes.

    Enforces the creation of some fields in the pass while allowing passes to
    override ``__init__``.
    """

    # Drop this functionality in the future.
    # This metaclass provides a pass equivalence evaluation based on the constructor arguments.
    # This implicit fake-hash based equivalence is fragile, and the pass developer must
    # explicitly implement equivalence check logic for each pass if necessary.
    # Currently, this metaclass is just here for backward compatibility, because
    # circuit pass manager has a functionality to avoid multiple execution of the
    # same pass (even though they are scheduled so). This is managed by the valid_passes set,
    # and executed passes are added to this collection to avoid future execution.
    # Dropping this metaclass causes many unittest failures and thus this is
    # considered as a breaking API change.
    # For example, test.python.transpiler.test_pass_scheduler.TestLogPasses.test_passes_in_linear

    def __call__(cls, *args, **kwargs):
        pass_instance = type.__call__(cls, *args, **kwargs)
        pass_instance._hash = hash(MetaPass._freeze_init_parameters(cls, args, kwargs))
        return pass_instance

    @staticmethod
    def _freeze_init_parameters(class_, args, kwargs):
        self_guard = object()
        init_signature = signature(class_.__init__)
        bound_signature = init_signature.bind(self_guard, *args, **kwargs)
        arguments = [("class_.__name__", class_.__name__)]
        for name, value in bound_signature.arguments.items():
            if value == self_guard:
                continue
            if isinstance(value, Hashable):
                arguments.append((name, type(value), value))
            else:
                arguments.append((name, type(value), repr(value)))
        return frozenset(arguments)

    def mro(cls) -> list[type]:
        # Avoid metaclass conflict.
        mro_cls = super().mro()
        if GenericPass not in mro_cls:
            mro_cls.pop()
            mro_cls += [GenericPass, Task, object]
        return mro_cls


class BasePass(metaclass=MetaPass):
    """Base class for transpiler passes."""

    def __init__(self):
        super().__init__()
        self.preserves: Iterable[GenericPass] = []
        self._hash = hash(None)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    @abstractmethod
    def run(self, dag: DAGCircuit):  # pylint: disable=arguments-differ
        """Run a pass on the DAGCircuit. This is implemented by the pass developer.

        Args:
            dag: the dag on which the pass is run.

        Raises:
            NotImplementedError: when this is left unimplemented for a pass.
        """
        raise NotImplementedError

    @property
    def is_transformation_pass(self):
        """Check if the pass is a transformation pass.

        If the pass is a TransformationPass, that means that the pass can manipulate the DAG,
        but cannot modify the property set (but it can be read).
        """
        return isinstance(self, TransformationPass)

    @property
    def is_analysis_pass(self):
        """Check if the pass is an analysis pass.

        If the pass is an AnalysisPass, that means that the pass can analyze the DAG and write
        the results of that analysis in the property set. Modifications on the DAG are not allowed
        by this kind of pass.
        """
        return isinstance(self, AnalysisPass)

    def __call__(
        self,
        circuit: QuantumCircuit,
        property_set: PropertySet | dict | None = None,
    ) -> QuantumCircuit:
        """Runs the pass on circuit.

        Args:
            circuit: The dag on which the pass is run.
            property_set: Input/output property set. An analysis pass
                might change the property set in-place.

        Returns:
            If on transformation pass, the resulting QuantumCircuit.
            If analysis pass, the input circuit.
        """
        property_set_ = None
        if isinstance(property_set, dict):  # this includes (dict, PropertySet)
            property_set_ = PropertySet(property_set)

        if isinstance(property_set_, PropertySet):
            # pylint: disable=attribute-defined-outside-init
            self.property_set = property_set_

        result = self.run(circuit_to_dag(circuit))

        result_circuit = circuit

        if isinstance(property_set, dict):  # this includes (dict, PropertySet)
            property_set.clear()
            property_set.update(self.property_set)

        if isinstance(result, DAGCircuit):
            result_circuit = dag_to_circuit(result, copy_operations=False)
        elif result is None:
            result_circuit = circuit.copy()

        if self.property_set["layout"]:
            result_circuit._layout = TranspileLayout(
                initial_layout=self.property_set["layout"],
                input_qubit_mapping=self.property_set["original_qubit_indices"],
                final_layout=self.property_set["final_layout"],
            )
        if self.property_set["clbit_write_latency"] is not None:
            result_circuit._clbit_write_latency = self.property_set["clbit_write_latency"]
        if self.property_set["conditional_latency"] is not None:
            result_circuit._conditional_latency = self.property_set["conditional_latency"]
        if self.property_set["node_start_time"]:
            # This is dictionary keyed on the DAGOpNode, which is invalidated once
            # dag is converted into circuit. So this schedule information is
            # also converted into list with the same ordering with circuit.data.
            topological_start_times = []
            start_times = self.property_set["node_start_time"]
            for dag_node in result.topological_op_nodes():
                topological_start_times.append(start_times[dag_node])
            result_circuit._op_start_times = topological_start_times

        return result_circuit


class AnalysisPass(BasePass):  # pylint: disable=abstract-method
    """An analysis pass: change property set, not DAG."""

    def execute(
        self,
        passmanager_ir: DAGCircuit,
        property_set: PropertySet | None = None,
        state: PassState | None = None,
        callback: Callable = None,
    ) -> DAGCircuit:
        """Execute optimization task for input passmanager IR.

        Args:
            passmanager_ir: Passmanager IR to optimize.
            property_set: A mutable data collection shared among all tasks.
            state: A local state information associated with this optimization workflow.
            callback: A callback function which is caller per execution of optimization task.

        Returns:
            Optimized passmanager IR.
        """
        # analysis pass must not mutate DAG object
        fenced_dag = FencedDAGCircuit(passmanager_ir)

        # analysis pass returns nothing, just mutates the property set
        super().execute(
            passmanager_ir=fenced_dag,
            property_set=property_set,
            state=state,
            callback=callback,
        )

        return passmanager_ir


class TransformationPass(BasePass):  # pylint: disable=abstract-method
    """A transformation pass: change DAG, not property set."""

    def execute(
        self,
        passmanager_ir: DAGCircuit,
        property_set: PropertySet | None = None,
        state: PassState | None = None,
        callback: Callable = None,
    ) -> DAGCircuit:
        """Execute optimization task for input passmanager IR.

        Args:
            passmanager_ir: Passmanager IR to optimize.
            property_set: A mutable data collection shared among all tasks.
            state: A local state information associated with this optimization workflow.
            callback: A callback function which is caller per execution of optimization task.

        Returns:
            Optimized passmanager IR.
        """
        new_dag = super().execute(
            passmanager_ir=passmanager_ir,
            property_set=property_set,
            state=state,
            callback=callback,
        )

        if self.state.previous_run == RunState.SUCCESS:
            self.state.completed_passes.intersection_update(set(self.preserves))

            if isinstance(new_dag, DAGCircuit):
                # Copy calibration data from the original program
                new_dag.calibrations = passmanager_ir.calibrations
            else:
                raise TranspilerError(
                    "Transformation passes should return a transformed dag."
                    f"The pass {self.__class__.__name__} is returning a {type(new_dag)}"
                )

        return new_dag
