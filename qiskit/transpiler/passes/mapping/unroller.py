# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Pass for unrolling a circuit to a given basis."""

import networkx as nx

from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit.transpiler._basepasses import TransformationPass


class Unroller(TransformationPass):
    """
    Unroll (expand) non-basis, non-opaque instructions recursively
    to a desired basis, using decomposition rules defined for each instruction.
    """

    def __init__(self, basis=None):
        """
        Args:
            basis (list[Instruction]): target basis gates to unroll to
        """
        super().__init__()
        self.basis = basis or []
        self.basis += ['U', 'CX']  # Add default basis.

    def run(self, dag):
        """Expand all op nodes to the given basis.

        If self.basis is empty, the circuit is unrolled down to
        fundamental (opaque) gates (U, CX).

        Args:
            dag(DAGCircuit): input dag

        Returns:
            DAGCircuit: output unrolled dag

        Raises:
            TranspilerError: if no decomposition rule is found for an op
        """
        # Walk through the DAG and expand each non-basis node
        for node in dag.get_gate_nodes():
            current_node = dag.multi_graph.node[node]

            if current_node["op"].name in self.basis:  # If already a base, ignore.
                continue

            decomposition_rules = current_node["op"].decompositions()

            # TODO: allow choosing other possible decompositions
            decomposition_dag = self.run(decomposition_rules[0])  # recursively unroll gates

            condition = current_node["condition"]
            # the decomposition rule must be amended if used in a
            # conditional context. delete the op nodes and replay
            # them with the condition.
            if condition:
                decomposition_dag.add_creg(condition[0])
                to_replay = []
                for n_it in nx.topological_sort(decomposition_dag.multi_graph):
                    n = decomposition_dag.multi_graph.nodes[n_it]
                    if n["type"] == "op":
                        n["op"].control = condition
                        to_replay.append(n)
                for n in decomposition_dag.get_op_nodes():
                    decomposition_dag._remove_op_node(n)
                for n in to_replay:
                    decomposition_dag.apply_operation_back(n["op"], condition=condition)

            # the wires for substitute_circuit_one are expected as qargs first,
            # then cargs, then conditions
            qwires = [w for w in decomposition_dag.wires
                      if isinstance(w[0], QuantumRegister)]
            cwires = [w for w in decomposition_dag.wires
                      if isinstance(w[0], ClassicalRegister)]

            dag.substitute_circuit_one(node,
                                       decomposition_dag,
                                       qwires + cwires)
        return dag
