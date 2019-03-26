# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Pass for decomposing 3q (or more) gates into 2q or 1q gates."""

from qiskit.transpiler.basepasses import TransformationPass


class Unroll3qOrMore(TransformationPass):
    """
    Recursively expands 3+ qubit gates until the circuit only contains
    1 qubit and 2qubit gates.
    """

    def run(self, dag):
        """Expand 3+ qubit gates using their decomposition rules.

        Args:
            dag(DAGCircuit): input dag
        Returns:
            DAGCircuit: output dag with maximum node degrees of 2
        """
        for node in dag.threeQ_or_more_nodes():
            decomposition_rules = node.op.decompositions()

            # TODO: allow choosing other possible decompositions
            decomposition_dag = self.run(decomposition_rules[0])  # recursively unroll

            dag.substitute_node_with_dag(node, decomposition_dag)
        return dag
