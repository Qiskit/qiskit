# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Pass for decompose 3q gates into 2q or 1q gates."""

from qiskit.transpiler._basepasses import TransformationPass


class Decompose3Q(TransformationPass):
    """
    Recursively expands 3q gates using their decomposition rules.
    """

    def run(self, dag):
        """Expand 3q gates using their decomposition rules.

        Args:
            dag(DAGCircuit): input dag
        Returns:
            DAGCircuit: output dag where 3q gates were expanded.
        """
        for node_id, node_data in dag.get_3q_nodes():
            decomposition_rules = node_data["op"].decompositions()

            # TODO: allow choosing other possible decompositions
            decomposition_dag = self.run(decomposition_rules[0])  # recursively decompose 3q gates

            dag.substitute_node_with_dag(node_id, decomposition_dag)
        return dag
