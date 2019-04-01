# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""
A pass that merges any adjacent barriers into one
"""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard.barrier import Barrier


class MergeAdjacentBarriers(TransformationPass):
    """Returns a circuit with any adjacent barriers merged together"""

    def run(self, dag):

        # sorted to so that they are in the order they were added to the DAG
        # so ancestors/descendants makes sense
        barriers = sorted(dag.named_nodes('barrier'))

        # get dict of barrier merges
        node_to_barrier = MergeAdjacentBarriers._collect_potential_merges(dag, barriers)

        if not node_to_barrier:
            return dag

        # add the merged barriers to a new DAG
        new_dag = DAGCircuit()

        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        # go over current nodes, and add them to the new dag
        for node in dag.nodes_in_topological_order():

            if node.type == 'op':
                if node.name == 'barrier':
                    if node in node_to_barrier:
                        new_dag.apply_operation_back(node_to_barrier[node])
                else:
                    # copy the condition over too
                    if node.condition:
                        new_dag.apply_operation_back(node.op, condition=node.condition)
                    else:
                        new_dag.apply_operation_back(node.op)
        return new_dag

    @staticmethod
    def _collect_potential_merges(dag, barriers):
        """
        Returns a dict of DAGNode : Barrier objects, where the barrier needs to be
        inserted where the corresponding DAGNode appears in the main DAG
        """
        # if only got 1 or 0 barriers then can't merge
        if len(barriers) < 2:
            return None

        # mapping from the node that will be the main barrier to the
        # barrier object that gets built up
        node_to_barrier = {}

        # Start from the first barrier
        current_barrier = barriers[0]
        start_of_barrier = current_barrier

        current_qubits = set(current_barrier.qargs)
        current_ancestors = dag.ancestors(current_barrier)
        current_descendants = dag.descendants(current_barrier)

        barrier_to_add = Barrier(qubits=current_qubits)

        for next_barrier in barriers[1:]:

            next_ancestors = dag.ancestors(next_barrier)
            next_descendants = dag.descendants(next_barrier)
            next_qubits = set(next_barrier.qargs)

            if (
                    not current_qubits.isdisjoint(next_qubits)
                    and current_ancestors.isdisjoint(next_descendants)
                    and current_descendants.isdisjoint(next_ancestors)
            ):

                # can be merged
                current_ancestors = current_ancestors | next_ancestors
                current_descendants = current_descendants | next_descendants
                current_qubits = current_qubits | next_qubits

                # update the barrier that will be added back to include this barrier
                barrier_to_add = Barrier(qubits=current_qubits)

            else:
                # store the previously made barrier
                if barrier_to_add:
                    node_to_barrier[start_of_barrier] = barrier_to_add

                # reset the properties
                current_qubits = set(next_barrier.qargs)
                current_ancestors = dag.ancestors(next_barrier)
                current_descendants = dag.descendants(next_barrier)

                barrier_to_add = Barrier(qubits=current_qubits)
                start_of_barrier = next_barrier

        if barrier_to_add:
            node_to_barrier[start_of_barrier] = barrier_to_add

        return node_to_barrier
