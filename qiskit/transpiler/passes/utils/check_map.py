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

"""Check if a DAG circuit is already mapped to a coupling map."""

from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.circuit.controlflow import ControlFlowOp


class CheckMap(AnalysisPass):
    """Check if a DAG circuit is already mapped to a coupling map.

    Check if a DAGCircuit is mapped to `coupling_map` by checking that all
    2-qubit interactions are laid out to be physically close, setting the
    property ``is_swap_mapped`` to ``True`` or ``False`` accordingly.
    """

    def __init__(self, coupling_map):
        """CheckMap initializer.

        Args:
            coupling_map (CouplingMap): Directed graph representing a coupling map.
        """
        super().__init__()
        self.coupling_map = coupling_map

    def run(self, dag):
        """Run the CheckMap pass on `dag`.

        If `dag` is mapped to `coupling_map`, the property
        `is_swap_mapped` is set to True (or to False otherwise).

        Args:
            dag (DAGCircuit): DAG to map.
        """
        from qiskit.converters import circuit_to_dag

        self.property_set["is_swap_mapped"] = True

        if self.coupling_map is None or len(self.coupling_map.graph) == 0:
            return

        qubit_indices = {bit: index for index, bit in enumerate(dag.qubits)}
        # Use dist matrix directly to avoid validation overhead
        dist_matrix = self.coupling_map.distance_matrix
        for node in dag.op_nodes(include_directives=False):
            is_controlflow_op = isinstance(node.op, ControlFlowOp)
            if len(node.qargs) == 2 and not is_controlflow_op:
                if dag.has_calibration_for(node):
                    continue
                physical_q0 = qubit_indices[node.qargs[0]]
                physical_q1 = qubit_indices[node.qargs[1]]
                if dist_matrix[physical_q0, physical_q1] != 1:
                    self.property_set["check_map_msg"] = "{}({}, {}) failed".format(
                        node.name,
                        physical_q0,
                        physical_q1,
                    )
                    self.property_set["is_swap_mapped"] = False
                    return
            elif is_controlflow_op:
                order = [qubit_indices[bit] for bit in node.qargs]
                for block in node.op.blocks:
                    dag_block = circuit_to_dag(block)
                    mapped_dag = dag.copy_empty_like()
                    mapped_dag.compose(dag_block, qubits=order)
                    self.run(mapped_dag)
                    if not self.property_set["is_swap_mapped"]:
                        return
