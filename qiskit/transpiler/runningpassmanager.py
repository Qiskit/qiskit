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

import logging
from time import time
from typing import Dict

from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager.base_pass_runner import BasePassRunner
from qiskit.passmanager.exceptions import PassManagerError
from qiskit.passmanager.fencedobjs import FencedObject
from qiskit.transpiler.basepasses import BasePass

from .exceptions import TranspilerError
from .layout import TranspileLayout

logger = logging.getLogger(__name__)


class RunningPassManager(BasePassRunner):
    """A RunningPassManager is a running pass manager."""

    def __init__(self, max_iteration: int):
        super().__init__(max_iteration)

        self.callback = None
        self.count = 0
        self.output_name = None

    def append(self, passes, **flow_controller_conditions):
        try:
            super().append(passes, **flow_controller_conditions)
        except PassManagerError as ex:
            # Override error for backward compatibility.
            raise TranspilerError(ex.message) from ex

    def _to_passmanager_ir(self, in_program):
        return circuit_to_dag(in_program)

    def _to_target(self, passmanager_ir):
        target = dag_to_circuit(passmanager_ir)
        target.name = self.output_name

        if self.property_set["layout"] is not None:
            target._layout = TranspileLayout(
                initial_layout=self.property_set["layout"],
                input_qubit_mapping=self.property_set["original_qubit_indices"],
                final_layout=self.property_set["final_layout"],
            )
        target._clbit_write_latency = self.property_set["clbit_write_latency"]
        target._conditional_latency = self.property_set["conditional_latency"]

        if self.property_set["node_start_time"]:
            # This is dictionary keyed on the DAGOpNode, which is invalidated once
            # dag is converted into circuit. So this schedule information is
            # also converted into list with the same ordering with circuit.data.
            topological_start_times = []
            start_times = self.property_set["node_start_time"]
            for dag_node in passmanager_ir.topological_op_nodes():
                topological_start_times.append(start_times[dag_node])
            target._op_start_times = topological_start_times

        return target

    # pylint: disable=arguments-differ
    def run(self, circuit, output_name=None, callback=None):
        """Run all the passes on a QuantumCircuit

        Args:
            circuit (QuantumCircuit): circuit to transform via all the registered passes
            output_name (str): The output circuit name. If not given, the same as the
                               input circuit
            callback (callable): A callback function that will be called after each pass execution.

        Returns:
            QuantumCircuit: Transformed circuit.
        """
        if callback:
            self.callback = callback

        if output_name:
            self.output_name = output_name
        else:
            self.output_name = circuit.name

        return super().run(in_program=circuit)

    def _do_atomic_pass(
        self,
        pass_: BasePass,
        passmanager_ir: DAGCircuit,
        options: Dict,
    ) -> DAGCircuit:
        """Do an atomic pass.

        Args:
            pass_ : Pass to do.
            passmanager_ir: The dag on which the pass is ran.
            options: PassManager options.

        Returns:
            The transformed dag in case of a transformation pass.
            The same input dag in case of an analysis pass.

        Raises:
            TranspilerError: If the pass is not a proper pass instance.
        """
        # First, do the requires of pass_
        for required_pass in pass_.requires:
            passmanager_ir = self._do_atomic_pass(required_pass, passmanager_ir, options)

        # Run the pass itself, if not already run
        if pass_ not in self.valid_passes:
            passmanager_ir = self._run_this_pass(pass_, passmanager_ir)

            # update the valid_passes property
            self._update_valid_passes(pass_)

        return passmanager_ir

    def _run_this_pass(self, pass_, dag):
        pass_.property_set = self.property_set
        if pass_.is_transformation_pass:
            # Measure time if we have a callback or logging set
            start_time = time()
            new_dag = pass_.run(dag)
            end_time = time()
            run_time = end_time - start_time
            # Execute the callback function if one is set
            if self.callback:
                self.callback(
                    pass_=pass_,
                    dag=new_dag,
                    time=run_time,
                    property_set=self.property_set,
                    count=self.count,
                )
                self.count += 1
            self._log_pass(start_time, end_time, pass_.name())
            if isinstance(new_dag, DAGCircuit):
                new_dag.calibrations = dag.calibrations
            else:
                raise TranspilerError(
                    "Transformation passes should return a transformed dag."
                    "The pass %s is returning a %s" % (type(pass_).__name__, type(new_dag))
                )
            dag = new_dag
        elif pass_.is_analysis_pass:
            # Measure time if we have a callback or logging set
            start_time = time()
            pass_.run(FencedDAGCircuit(dag))
            end_time = time()
            run_time = end_time - start_time
            # Execute the callback function if one is set
            if self.callback:
                self.callback(
                    pass_=pass_,
                    dag=dag,
                    time=run_time,
                    property_set=self.property_set,
                    count=self.count,
                )
                self.count += 1
            self._log_pass(start_time, end_time, pass_.name())
        else:
            raise TranspilerError("I dont know how to handle this type of pass")
        return dag

    def _log_pass(self, start_time, end_time, name):
        log_msg = f"Pass: {name} - {(end_time - start_time) * 1000:.5f} (ms)"
        logger.info(log_msg)


def __getattr__(name):
    # For backward compatibility. Flow controllers are moved to pass manager module.
    from qiskit.passmanager import flow_controller

    return getattr(flow_controller, name)
