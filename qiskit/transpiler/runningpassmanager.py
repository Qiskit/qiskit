# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""RunningPassManager class for the transpiler.
This object holds the state of a pass manager during running-time."""
from __future__ import annotations
import logging
import inspect
from functools import partial, wraps
from typing import Callable

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager import BasePassRunner
from qiskit.passmanager.flow_controllers import (
    PassSequence,
    FlowController,
    DoWhileController,
    ConditionalController,
)
from qiskit.passmanager.exceptions import PassManagerError
from qiskit.transpiler.basepasses import BasePass
from .exceptions import TranspilerError
from .fencedobjs import FencedPropertySet, FencedDAGCircuit
from .layout import TranspileLayout

logger = logging.getLogger(__name__)


class RunningPassManager(BasePassRunner):
    """A RunningPassManager is a running pass manager."""

    IN_PROGRAM_TYPE = QuantumCircuit
    OUT_PROGRAM_TYPE = QuantumCircuit
    IR_TYPE = DAGCircuit

    def __init__(self, max_iteration: int):
        """Initialize an empty PassManager object (with no passes scheduled).

        Args:
            max_iteration: The schedule looping iterates until the condition is met or until
                max_iteration is reached.
        """
        super().__init__(max_iteration)
        self.fenced_property_set = FencedPropertySet(self.property_set)

    def append(
        self,
        passes: PassSequence,
        **flow_controller_conditions,
    ):
        """Append a passes to the schedule of passes.

        Args:
            passes: passes to be added to schedule
            flow_controller_conditions: See add_flow_controller(): Dictionary of
            control flow plugins. Default:

                * do_while (callable property_set -> boolean): The passes repeat until the
                  callable returns False.
                  Default: `lambda x: False # i.e. passes run once`

                * condition (callable property_set -> boolean): The passes run only if the
                  callable returns True.
                  Default: `lambda x: True # i.e. passes run`
        """
        # attaches the property set to the controller so it has access to it.
        if isinstance(passes, ConditionalController):
            passes.condition = partial(passes.condition, self.fenced_property_set)
        elif isinstance(passes, DoWhileController):
            if not isinstance(passes.do_while, partial):
                passes.do_while = partial(passes.do_while, self.fenced_property_set)
        else:
            flow_controller_conditions = self._normalize_flow_controller(flow_controller_conditions)
            passes = FlowController.controller_factory(
                passes, self.passmanager_options, **flow_controller_conditions
            )
        super().append(passes)

    def _normalize_flow_controller(self, flow_controller):
        for name, param in flow_controller.items():
            if callable(param):
                flow_controller[name] = partial(param, self.fenced_property_set)
            else:
                raise TranspilerError("The flow controller parameter %s is not callable" % name)
        return flow_controller

    def _to_passmanager_ir(self, in_program: QuantumCircuit) -> DAGCircuit:
        if not isinstance(in_program, QuantumCircuit):
            raise TranspilerError(f"Input {in_program.__class__} is not QuantumCircuit.")
        return circuit_to_dag(in_program)

    def _to_target(self, passmanager_ir: DAGCircuit) -> QuantumCircuit:
        if not isinstance(passmanager_ir, DAGCircuit):
            raise TranspilerError(f"Input {passmanager_ir.__class__} is not DAGCircuit.")

        circuit = dag_to_circuit(passmanager_ir, copy_operations=False)
        circuit.name = self.metadata["output_name"]

        if self.property_set["layout"] is not None:
            circuit._layout = TranspileLayout(
                initial_layout=self.property_set["layout"],
                input_qubit_mapping=self.property_set["original_qubit_indices"],
                final_layout=self.property_set["final_layout"],
            )
        circuit._clbit_write_latency = self.property_set["clbit_write_latency"]
        circuit._conditional_latency = self.property_set["conditional_latency"]

        if self.property_set["node_start_time"]:
            # This is dictionary keyed on the DAGOpNode, which is invalidated once
            # dag is converted into circuit. So this schedule information is
            # also converted into list with the same ordering with circuit.data.
            topological_start_times = []
            start_times = self.property_set["node_start_time"]
            for dag_node in passmanager_ir.topological_op_nodes():
                topological_start_times.append(start_times[dag_node])
            circuit._op_start_times = topological_start_times

        return circuit

    # pylint: disable=arguments-differ
    def run(
        self,
        circuit: QuantumCircuit,
        output_name: str = None,
        callback: Callable = None,
    ) -> QuantumCircuit:
        """Run all the passes on a QuantumCircuit

        Args:
            circuit: Circuit to transform via all the registered passes.
            output_name: The output circuit name. If not given, the same as the input circuit.
            callback: A callback function that will be called after each pass execution.

        Returns:
            QuantumCircuit: Transformed circuit.
        """
        return super().run(
            in_program=circuit,
            callback=_rename_callback_args(callback),
            output_name=output_name or circuit.name,
        )

    def _run_base_pass(
        self,
        pass_: BasePass,
        passmanager_ir: DAGCircuit,
    ) -> DAGCircuit:
        """Do either a pass and its "requires" or FlowController.

        Args:
            pass_: A base pass to run.
            passmanager_ir: Pass manager IR, i.e. DAGCircuit for this class.

        Returns:
            The transformed dag in case of a transformation pass.
            The same input dag in case of an analysis pass.

        Raises:
            TranspilerError: When transform pass returns non DAGCircuit.
            TranspilerError: When pass is neither transform pass nor analysis pass.
        """
        pass_.property_set = self.property_set

        if pass_.is_transformation_pass:
            # Measure time if we have a callback or logging set
            new_dag = pass_.run(passmanager_ir)
            if isinstance(new_dag, DAGCircuit):
                new_dag.calibrations = passmanager_ir.calibrations
            else:
                raise TranspilerError(
                    "Transformation passes should return a transformed dag."
                    "The pass %s is returning a %s" % (type(pass_).__name__, type(new_dag))
                )
            passmanager_ir = new_dag
        elif pass_.is_analysis_pass:
            # Measure time if we have a callback or logging set
            pass_.run(FencedDAGCircuit(passmanager_ir))
        else:
            raise TranspilerError("I dont know how to handle this type of pass")
        return passmanager_ir

    def _update_valid_passes(self, pass_):
        super()._update_valid_passes(pass_)
        if not pass_.is_analysis_pass:  # Analysis passes preserve all
            self.valid_passes.intersection_update(set(pass_.preserves))


def _rename_callback_args(callback):
    """A helper function to run callback with conventional argument names."""
    if callback is None:
        return callback

    def _call_with_dag(pass_, passmanager_ir, time, property_set, count):
        callback(
            pass_=pass_,
            dag=passmanager_ir,
            time=time,
            property_set=property_set,
            count=count,
        )

    return _call_with_dag


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


for _name, _method in inspect.getmembers(RunningPassManager, predicate=inspect.isfunction):
    if _name.startswith("_"):
        # Ignore protected and private.
        # User usually doesn't directly execute and catch error from these methods.
        continue
    _wrapped = _replace_error(_method)
    setattr(RunningPassManager, _name, _wrapped)
