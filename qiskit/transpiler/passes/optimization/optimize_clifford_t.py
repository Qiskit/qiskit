# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Combine consecutive T/Tdg gates in a Clifford+T circuit."""

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.library import SGate, SdgGate


class OptimizeCliffordT(TransformationPass):
    """An optimization pass for Clifford+T circuits.

    Currently all the pass does is merging pairs of consecutive T-gates into
    S-gates, and pair of consecutive Tdg-gates into Sdg-gates.
    """

    def run(self, dag: DAGCircuit):
        """
        Run the OptimizeCliffordT pass on `dag`.

        Args:
            dag: The directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """

        new_dag = dag.copy_empty_like()

        nodes = list(dag.topological_op_nodes())
        num_nodes = len(nodes)
        idx = 0

        while idx < num_nodes - 1:
            cur_node = nodes[idx]
            next_node = nodes[idx + 1]
            if cur_node.name == "t" and next_node.name == "t" and cur_node.qargs == next_node.qargs:
                # Combine two consecutive T-gates into an S-gate
                new_dag.apply_operation_back(SGate(), cur_node.qargs, cur_node.cargs)
                idx += 2
            elif (
                nodes[idx].name == "tdg"
                and nodes[idx + 1].name == "tdg"
                and nodes[idx].qargs == nodes[idx + 1].qargs
            ):
                # Combine two consecutive Tdg-gates into an Sdg-gate
                new_dag.apply_operation_back(SdgGate(), cur_node.qargs, cur_node.cargs)
                idx += 2
            else:
                new_dag.apply_operation_back(cur_node.op, cur_node.qargs, cur_node.cargs)
                idx += 1

        # Handle the last element (if any)
        if idx == num_nodes - 1:
            cur_node = nodes[idx]
            new_dag.apply_operation_back(cur_node.op, cur_node.qargs, cur_node.cargs)

        return new_dag
