# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
This pass adds a barrier before the set of final measurements. Measurements
are considered final if they are followed by no other operations (aside from
other measurements or barriers.)

A new barrier will not be added if an equivalent barrier is already present.
"""

from qiskit.extensions.standard.barrier import Barrier
from qiskit.transpiler._basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit


class BarrierBeforeFinalMeasurements(TransformationPass):
    """Adds a barrier before final measurements."""

    def run(self, dag):
        """Return a circuit with a barrier before last measurments."""

        # Collect DAG nodes which are followed only by barriers or other measures.
        final_op_types = ['measure', 'barrier']
        final_ops = []
        for candidate_op in dag.named_nodes(*final_op_types):
            is_final_op = True
            for _, child_successors in dag.bfs_successors(candidate_op):
                if any(dag.multi_graph.node[suc]['type'] == 'op' and
                       dag.multi_graph.node[suc]['op'].name not in final_op_types
                       for suc in child_successors):
                    is_final_op = False
                    break

            if is_final_op:
                final_ops.append(candidate_op)

        if not final_ops:
            return dag

        # Create a layer with the barrier and add registers from the original dag.
        barrier_layer = DAGCircuit()
        for qreg in dag.qregs.values():
            barrier_layer.add_qreg(qreg)
        for creg in dag.cregs.values():
            barrier_layer.add_creg(creg)

        final_qubits = set(dag.multi_graph.node[final_op]['qargs'][0]
                           for final_op in final_ops)

        barrier_layer.apply_operation_back(Barrier(qubits=final_qubits))
        new_barrier_id = barrier_layer.node_counter

        # Preserve order of final ops collected earlier from the original DAG.
        ordered_node_ids = [node_id for node_id in dag.node_nums_in_topological_order()
                            if node_id in set(final_ops)]
        ordered_final_nodes = [dag.multi_graph.node[node] for node in ordered_node_ids]

        # Move final ops to the new layer and append the new layer to the DAG.
        for final_node in ordered_final_nodes:
            barrier_layer.apply_operation_back(final_node['op'])

        for final_op in final_ops:
            dag._remove_op_node(final_op)

        # Check to see if the new barrier added to the DAG is equivalent to any
        # existing barriers, and if so, consolidate the two.
        our_ancestors = barrier_layer.ancestors(new_barrier_id)
        our_descendants = barrier_layer.descendants(new_barrier_id)
        our_qubits = final_qubits

        existing_barriers = barrier_layer.named_nodes('barrier')
        existing_barriers.remove(new_barrier_id)

        for candidate_barrier in existing_barriers:
            their_ancestors = barrier_layer.ancestors(candidate_barrier)
            their_descendants = barrier_layer.descendants(candidate_barrier)
            their_qubits = set(barrier_layer.multi_graph.nodes[candidate_barrier]['op'].qargs)

            if (
                    not our_qubits.isdisjoint(their_qubits)
                    and our_ancestors.isdisjoint(their_descendants)
                    and our_descendants.isdisjoint(their_ancestors)
            ):
                merge_barrier = Barrier(qubits=(our_qubits | their_qubits))
                barrier_layer.apply_operation_front(merge_barrier)

                merge_barrier_id = barrier_layer.node_counter
                our_ancestors = our_ancestors | their_ancestors
                our_descendants = our_descendants | their_descendants

                barrier_layer._remove_op_node(candidate_barrier)
                barrier_layer._remove_op_node(new_barrier_id)

                new_barrier_id = merge_barrier_id

        dag.extend_back(barrier_layer)

        return dag
