# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Pass for unrolling a circuit to a given basis."""

from qiskit.transpiler._basepasses import TransformationPass
from qiskit import QiskitError


class Unroller(TransformationPass):
    """
    Unroll (expand) non-basis, non-opaque instructions recursively
    to a desired basis, using decomposition rules defined for each instruction.
    """

    def __init__(self, basis):
        """
        Args:
            basis (list[qiskit.circuit.gate.Gate]): Target basis gates to unroll to.
        """
        super().__init__()
        self.basis = basis

    def run(self, dag):
        """Expand all op nodes to the given basis.

        If self.basis is empty, the circuit is unrolled down to
        fundamental (opaque) gates (U, CX).

        Args:
            dag(DAGCircuit): input dag

        Raises:
            QiskitError: if unable to unroll given the basis due to undefined
            decomposition rules (such as a bad basis) or excessive recursion.

        Returns:
            DAGCircuit: output unrolled dag
        """
        # Walk through the DAG and expand each non-basis node
        for node in dag.gate_nodes():
            current_node = dag.multi_graph.node[node]
            if current_node["op"].name in self.basis:  # If already a base, ignore.
                continue

            # TODO: allow choosing other possible decompositions
            try:
                decomposition_rules = current_node["op"].decompositions()
            except NotImplementedError:
                raise QiskitError("Cannot unroll the circuit to the given basis, "
                                  + str(self.basis) +
                                  ". No decomposition rules defined for "
                                  + current_node["op"].name + ".")
            except RecursionError:
                raise QiskitError("Cannot unroll the circuit to the given basis, "
                                  + str(self.basis) +
                                  ", due to infinite recursion at node "
                                  + current_node["op"].name + ":" + str(self.basis))

            if not decomposition_rules:
                raise QiskitError("Cannot unroll the circuit to the given basis, "
                                  + str(self.basis) + ". " +
                                  current_node["op"].name +
                                  " is defined in terms of an invalid basis.")

            decomposition_dag = self.run(decomposition_rules[0])  # recursively unroll gates
            dag.substitute_node_with_dag(node, decomposition_dag)
        return dag
