# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""This pass collapses 2-qubit gates that commute into :class:`.Commuting2qBlock`."""


from qiskit.circuit import Qubit, Gate
from qiskit.converters import dag_to_dagdependency, dagdependency_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.passes.optimization.collect_blocks import BlockSplitter, BlockCollector
from qiskit.transpiler.passes.optimization.commutation_analysis import CommutationAnalysis
from qiskit.transpiler import TransformationPass, TranspilerError

from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import (
    Commuting2qBlock,
)


class Commuting2qGateGrouper(TransformationPass):
    """A pass that collapses 2-qubit gates that commute into :class:`.Commuting2qBlock`."""

    def __init__(self, split_blocks=True, min_nodes_per_block=2):
        """
        Args:
            split_blocks (bool): if True, splits detected blocks into sub-blocks over disjoint qubit sets.
            min_nodes_per_block (int): specifies the minimum number of nodes in a block on
                which to apply collapse_function.
        """
        self.split_blocks = split_blocks
        self.min_nodes_per_block = min_nodes_per_block

        super().__init__()

    def collapse_function(self, blocks, dag):
        """Specifies what to do with collected blocks.
        """
        # Replace every discovered block by a linear function
        global_index_map = {wire: idx for idx, wire in enumerate(dag.qubits)}
        for commuting_op_set in blocks:
            # Find the set of all qubits used in this block
            cur_qubits = set()
            for node in commuting_op_set:
                cur_qubits.update(node.qargs)

            # For reproducibility, order these qubits compatibly with the global order
            sorted_qubits = sorted(cur_qubits, key=lambda x: global_index_map[x])
            wire_pos_map = dict((qb, ix) for ix, qb in enumerate(sorted_qubits))

            op = Commuting2qBlock(commuting_op_set)

            # Replace the block by the constructed circuit
            # QUESTION: WHAT TO DO WITH CLASSICAL BITS?!
            dag.replace_block_with_op(commuting_op_set, op, wire_pos_map, cycle_check=False)

        return dag

    def run(self, dag):
        """Run the Commuting2qGateGrouper pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """

        def _is_2q(node):
            """Two-qubit gates."""
            return isinstance(node.op, Gate) and node.op.condition is None and len(node.qargs) == 2

        processed_dag = dag_to_dagdependency(dag)

        matching_blocks = BlockCollector(processed_dag).collect_all_commuting_blocks(filter_fn=_is_2q)

        if self.split_blocks:
            blocks = []
            for block in matching_blocks:
                blocks.extend(BlockSplitter().run(block))
        else:
            blocks = matching_blocks

        filtered_blocks = [block for block in blocks if len(block) >= self.min_nodes_per_block]

        collapsed_dep_dag = self.collapse_function(filtered_blocks, processed_dag)

        collapsed_dag = dagdependency_to_dag(collapsed_dep_dag)

        return collapsed_dag


