# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Pass for peep-hole cancellation of consecutive CX gates.
"""
from qiskit.transpiler._basepasses import TransformationPass


class CXCancellation(TransformationPass):
    """Cancel back-to-back 'cx' gates in dag."""

    def run(self, dag):
        """
        Run one pass of cx cancellation on the circuit

        Args:
            dag (DAGCircuit): the directed acyclic graph to run on.
        Returns:
            DAGCircuit: Transformed DAG.
        """
        cx_runs = dag.collect_runs(["cx"])
        for cx_run in cx_runs:
            # Partition the cx_run into chunks with equal gate arguments
            partition = []
            chunk = []
            for i in range(len(cx_run) - 1):
                chunk.append(cx_run[i])
                qargs0 = dag.multi_graph.node[cx_run[i]]["qargs"]
                qargs1 = dag.multi_graph.node[cx_run[i + 1]]["qargs"]
                if qargs0 != qargs1:
                    partition.append(chunk)
                    chunk = []
            chunk.append(cx_run[-1])
            partition.append(chunk)
            # Simplify each chunk in the partition
            for chunk in partition:
                if len(chunk) % 2 == 0:
                    for n in chunk:
                        dag._remove_op_node(n)
                else:
                    for n in chunk[1:]:
                        dag._remove_op_node(n)
        return dag
