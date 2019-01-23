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
        for candidate_op in dag.get_named_nodes(*final_op_types):
            nodes_after_candidate = [dag.multi_graph.nodes[node_id]
                                     for node_id in dag.descendants(candidate_op)]
            is_final_op = all([node['type'] == 'out'
                               or (node['type'] == 'op' and node['op'].name in final_op_types)
                               for node in nodes_after_candidate])

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

        final_qubits = []
        for final_op in final_ops:
            qubit = dag.multi_graph.node[final_op]['qargs'][0]
            if qubit not in final_qubits:
                final_qubits.append(qubit)

        dag.add_basis_element('barrier', len(final_qubits), 0, 0)
        barrier_layer.apply_operation_back(Barrier(qubits=final_qubits))

        # Preserve order of final ops collected earlier from the original DAG.
        ordered_node_ids = [node_id for node_id in dag.node_nums_in_topological_order()
                            if node_id in set(final_ops)]
        ordered_final_nodes = [dag.multi_graph.node[node] for node in ordered_node_ids]

        # If a barrier equivalent to the new barrier was already present in the
        # DAG, it would be first on the list of final nodes to re-add. Check if
        # it exists and return the DAG unmodified if so.
        barrier_to_add = barrier_layer.get_op_nodes(data=True)[0][1]
        if ordered_final_nodes[0] == barrier_to_add:
            return dag

        # Move final ops to the new layer and append the new layer to the DAG.
        for final_node in ordered_final_nodes:
            barrier_layer.apply_operation_back(final_node['op'])

        for final_op in final_ops:
            dag._remove_op_node(final_op)

        dag.extend_back(barrier_layer)

        return dag
