# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Pass for decompose a gate in a circuit."""

from qiskit.transpiler._basepasses import TransformationPass


class Decompose(TransformationPass):
    """
    Expand a gate in a circle using its decomposition rules.
    """

    def __init__(self, gate=None):
        """
        Args:
            gate (qiskit.circuit.gate.Gate): Gate to decompose.
        """
        super().__init__()
        self.gate = gate

    def run(self, dag):
        """Expand a given gate into its decomposition.

        Args:
            dag(DAGCircuit): input dag
        Returns:
            DAGCircuit: output dag where gate was expanded.
        """
        # Walk through the DAG and expand each non-basis node
        for node in dag.get_op_nodes(self.gate):
            current_node = dag.multi_graph.node[node]

            decomposition_rules = current_node["op"].decompositions()

            # TODO: allow choosing other possible decompositions
            decomposition_dag = decomposition_rules[0]

            dag.substitute_node_with_dag(node, decomposition_dag)
        return dag
