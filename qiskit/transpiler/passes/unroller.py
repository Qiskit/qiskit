# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Pass for unrolling a circuit to a given basis."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.exceptions import QiskitError


class Unroller(TransformationPass):
    """
    Unroll (expand) non-basis, non-opaque instructions recursively
    to a desired basis, using decomposition rules defined for each instruction.
    """

    def __init__(self, basis):
        """
        Args:
            basis (list[str]): Target basis gate names to unroll to, e.g. `['u3', 'cx']` .
        """
        super().__init__()
        self.basis = basis

    def run(self, dag):
        """Expand all op nodes to the given basis.


        Args:
            dag(DAGCircuit): input dag

        Raises:
            QiskitError: if unable to unroll given the basis due to undefined
            decomposition rules (such as a bad basis) or excessive recursion.

        Returns:
            DAGCircuit: output unrolled dag
        """
        # Walk through the DAG and expand each non-basis node
        for node_id, current_node in dag.gate_nodes(data=True):
            if current_node["op"].name in self.basis:  # If already a base, ignore.
                continue

            # TODO: allow choosing other possible decompositions
            decomposition_rules = current_node["op"].decompositions()

            if not decomposition_rules:
                raise QiskitError("Cannot unroll the circuit to the given basis, %s. "
                                  "The current node being expanded, %s, "
                                  "is defined in terms of an invalid basis." %
                                  (str(self.basis), current_node["op"].name))

            decomposition_dag = self.run(decomposition_rules[0])  # recursively unroll gates
            dag.substitute_node_with_dag(node_id, decomposition_dag)
        return dag
