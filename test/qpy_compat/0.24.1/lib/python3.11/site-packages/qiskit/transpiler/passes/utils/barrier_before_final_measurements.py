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


"""Add a barrier before final measurements."""

from qiskit.circuit.barrier import Barrier
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from .merge_adjacent_barriers import MergeAdjacentBarriers


class BarrierBeforeFinalMeasurements(TransformationPass):
    """Add a barrier before final measurements.

    This pass adds a barrier before the set of final measurements. Measurements
    are considered final if they are followed by no other operations (aside from
    other measurements or barriers.)
    """

    def run(self, dag):
        """Run the BarrierBeforeFinalMeasurements pass on `dag`."""
        # Collect DAG nodes which are followed only by barriers or other measures.
        final_op_types = ["measure", "barrier"]
        final_ops = []
        for candidate_node in dag.named_nodes(*final_op_types):
            is_final_op = True

            for _, child_successors in dag.bfs_successors(candidate_node):

                if any(
                    isinstance(suc, DAGOpNode) and suc.name not in final_op_types
                    for suc in child_successors
                ):
                    is_final_op = False
                    break

            if is_final_op:
                final_ops.append(candidate_node)

        if not final_ops:
            return dag

        # Create a layer with the barrier and add both bits and registers from the original dag.
        barrier_layer = DAGCircuit()
        barrier_layer.add_qubits(dag.qubits)
        for qreg in dag.qregs.values():
            barrier_layer.add_qreg(qreg)
        barrier_layer.add_clbits(dag.clbits)
        for creg in dag.cregs.values():
            barrier_layer.add_creg(creg)

        # Add a barrier across all qubits so swap mapper doesn't add a swap
        # from an unmeasured qubit after a measure.
        final_qubits = dag.qubits

        barrier_layer.apply_operation_back(Barrier(len(final_qubits)), list(final_qubits), [])

        # Preserve order of final ops collected earlier from the original DAG.
        ordered_final_nodes = [
            node for node in dag.topological_op_nodes() if node in set(final_ops)
        ]

        # Move final ops to the new layer and append the new layer to the DAG.
        for final_node in ordered_final_nodes:
            barrier_layer.apply_operation_back(final_node.op, final_node.qargs, final_node.cargs)

        for final_op in final_ops:
            dag.remove_op_node(final_op)

        dag.compose(barrier_layer)

        # Merge the new barrier into any other barriers
        adjacent_pass = MergeAdjacentBarriers()
        return adjacent_pass.run(dag)
