# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Replace each sequence of CX and SWAP gates by a single LinearFunction gate."""

from collections import deque
from qiskit.circuit.library.generalized_gates import LinearFunction
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit import QuantumCircuit
from qiskit.dagcircuit import DAGOpNode


def collect_linear_blocks(dag):
    """Collect blocks of linear gates."""

    blocks = []

    in_degree = dict()
    pending_linear_ops = deque()
    pending_non_linear_ops = deque()

    def is_linear(op):
        return op.name in ("cx", "swap") and op.condition is None

    def process_node(node):
        for suc in dag.successors(node):
            if not isinstance(suc, DAGOpNode):
                continue
            in_degree[suc] -= 1
            if in_degree[suc] > 0:
                continue

            if is_linear(suc.op):
                pending_linear_ops.append(suc)
            else:
                pending_non_linear_ops.append(suc)

    for node in dag.op_nodes():
        deg = sum(1 for op in dag.predecessors(node) if isinstance(op, DAGOpNode))
        if deg > 0:
            in_degree[node] = deg
            continue

        if is_linear(node.op):
            pending_linear_ops.append(node)
        else:
            pending_non_linear_ops.append(node)

    while pending_non_linear_ops or pending_linear_ops:

        # first collect as many non linear gates as possible
        while pending_non_linear_ops:
            node = pending_non_linear_ops.popleft()
            process_node(node)

        # now collect as many linear gates as possible
        cur_block = []
        while pending_linear_ops:
            node = pending_linear_ops.popleft()
            process_node(node)
            cur_block.append(node)

        if cur_block:
            blocks.append(cur_block)

    return blocks


def split_block_into_components(dag, block):
    """Given a block of gates, splits it into connected components."""

    if len(block) == 0:
        return block

    nodeset = set(block)
    colors = dict()
    color = -1

    def process_node_rec(node):
        """Given a node, recursively assigns the current color to all of its
        successors and predecessors."""

        if node not in colors.keys():
            colors[node] = color

            for pred in dag.predecessors(node):
                if node not in nodeset or not isinstance(pred, DAGOpNode):
                    continue
                process_node_rec(pred)

            for suc in dag.successors(node):
                if node not in nodeset or not isinstance(suc, DAGOpNode):
                    continue
                process_node_rec(suc)

    # Assign colors to nodes, so that nodes are in the same component iff they
    # have the same color
    for node in block:
        if node not in colors.keys():
            color = color + 1
            process_node_rec(node)

    # Split blocks based on color
    split_blocks = [[] for _ in range(color + 1)]
    for node in block:
        split_blocks[colors[node]].append(node)

    return split_blocks


def split_blocks_into_components(dag, blocks):
    """Given blocks of gates, splits each block into connected components,
    and returns a list of all blocks."""
    split_blocks = []
    for block in blocks:
        split_blocks.extend(split_block_into_components(dag, block))
    return split_blocks


class CollectLinearFunctions(TransformationPass):
    """Collect blocks of linear gates (:class:`.CXGate` and :class:`.SwapGate` gates)
    and replaces them by linear functions (:class:`.LinearFunction`)."""

    def run(self, dag):
        """Run the CollectLinearFunctions pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """

        # collect blocks of linear gates
        blocks = collect_linear_blocks(dag)

        # refine blocks by splitting into disconnected components
        blocks = split_blocks_into_components(dag, blocks)

        # Replace every discovered block by a linear function
        global_index_map = {wire: idx for idx, wire in enumerate(dag.qubits)}
        for cur_nodes in blocks:
            # Create linear functions only out of blocks with at least 2 gates
            if len(cur_nodes) == 1:
                continue

            # Find the set of all qubits used in this block
            cur_qubits = set()
            for node in cur_nodes:
                cur_qubits.update(node.qargs)

            # For reproducibility, order these qubits compatibly with the global order
            sorted_qubits = sorted(cur_qubits, key=lambda x: global_index_map[x])
            wire_pos_map = dict((qb, ix) for ix, qb in enumerate(sorted_qubits))

            # Construct a linear circuit
            qc = QuantumCircuit(len(cur_qubits))
            for node in cur_nodes:
                if node.op.name == "cx":
                    qc.cx(wire_pos_map[node.qargs[0]], wire_pos_map[node.qargs[1]])
                elif node.op.name == "swap":
                    qc.swap(wire_pos_map[node.qargs[0]], wire_pos_map[node.qargs[1]])

            # Create a linear function from this quantum circuit
            op = LinearFunction(qc)

            # Replace the block by the constructed circuit
            dag.replace_block_with_op(cur_nodes, op, wire_pos_map, cycle_check=False)

        return dag
