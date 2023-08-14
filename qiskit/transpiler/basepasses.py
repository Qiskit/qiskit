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

from abc import abstractmethod

from qiskit.passmanager.base_pass import GenericPass
from qiskit.passmanager.propertyset import PropertySet

from .layout import TranspileLayout


class BasePass(GenericPass):
    """Base class for transpiler passes."""

    @abstractmethod
    def run(self, dag):  # pylint: disable=arguments-differ
        """Run a pass on the DAGCircuit. This is implemented by the pass developer.

        Args:
            dag (DAGCircuit): the dag on which the pass is run.
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

    def __call__(self, circuit, property_set=None):
        """Runs the pass on circuit.

        Args:
            circuit (QuantumCircuit): the dag on which the pass is run.
            property_set (PropertySet or dict or None): input/output property set. An analysis pass
                might change the property set in-place.

        Returns:
            QuantumCircuit: If on transformation pass, the resulting QuantumCircuit. If analysis
                   pass, the input circuit.
        """
        from qiskit.converters import circuit_to_dag, dag_to_circuit
        from qiskit.dagcircuit.dagcircuit import DAGCircuit

        property_set_ = None
        if isinstance(property_set, dict):  # this includes (dict, PropertySet)
            property_set_ = PropertySet(property_set)

        if isinstance(property_set_, PropertySet):
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

    pass


class TransformationPass(BasePass):  # pylint: disable=abstract-method
    """A transformation pass: change DAG, not property set."""

    pass
