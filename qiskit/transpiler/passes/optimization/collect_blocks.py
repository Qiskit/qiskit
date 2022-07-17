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


"""Various ways to divide up DAG into blocks of nodes."""

from qiskit.dagcircuit import DAGOpNode
from qiskit.dagcircuit import DAGCircuit, DAGDependency


class BlockCollector:
    """Collecting blocks from DAGCircuit and DAGDependency.

    This class implements various strategies of dividing a DAG (direct acyclic graph)
    into blocks of nodes that satisfy certain criteria. It works both with the DAGCircuit
    representation of a DAG, and the DAGDependency representation of a DAG, where
    DagDependency takes into account commutativity between nodes. Collecting nodes
    from DAGCircuit generally leads to less optimal results, but should be faster,
    as it does not require to construct a DAGDependency beforehand. This may be useful
    with lower transpiler settings.


    Collecting blocks is generally not unique. The strategies explored here deal with
    heuristic approaches of the form 'starting from the input nodes of a DAG, collect
    the (heuristically) largest blocks of nodes that match certain criteria'.

    See Qiskit issue #5775 for additional details.
    """

    def __init__(self, dag):
        """
        Args:
            dag (DagCircuit): The input DAG (either DAGCircuit or DAGDependency).
        """

        self.dag = dag

        if isinstance(dag, DAGCircuit):
            self.is_dag_dependency = False

        elif isinstance(dag, DAGDependency):
            self.is_dag_dependency = True

        else:
            # ToDo: return an error
            assert False

        # For efficiency, we will compute (and keep updating) the in_degree for every node, that is
        # the number of the node's immediate predecessors. A node is a leaf (input) node iff its
        # in_degree is 0. When a node is (marked as) collected, the in_degrees of its immediate
        # successors are updated (by subtracting 1).
        # Additionally, we will explicitly keep the list of nodes with in_degree 0 as pending_nodes.
        self.pending_nodes = []
        self.in_degree = dict()
        for node in self._op_nodes():
            deg = len(self._direct_preds(node))
            self.in_degree[node] = deg
            if deg == 0:
                self.pending_nodes.append(node)

    def _op_nodes(self):
        """Returns DAG nodes
        (wrapper to handle both DagCircuit and DagDependency)."""
        if not self.is_dag_dependency:
            return self.dag.op_nodes()
        else:
            return self.dag.get_nodes()

    def _direct_preds(self, node):
        """Returns direct predecessors of a node
        (wrapper to handle both DagCircuit and DagDependency).
        """
        if not self.is_dag_dependency:
            return [pred for pred in self.dag.predecessors(node) if isinstance(pred, DAGOpNode)]
        else:
            return [
                self.dag.get_node(pred_id) for pred_id in self.dag.direct_predecessors(node.node_id)
            ]

    def _direct_succs(self, node):
        """Returns direct successors of a node
        (wrapper to handle both DagCircuit and DagDependency).
        """
        if not self.is_dag_dependency:
            return [succ for succ in self.dag.successors(node) if isinstance(succ, DAGOpNode)]
        else:
            return [
                self.dag.get_node(succ_id) for succ_id in self.dag.direct_successors(node.node_id)
            ]

    def have_uncollected_nodes(self):
        """Returns whether there are uncollected (pending) nodes"""
        return len(self.pending_nodes) > 0

    def _mark_node_as_collected(self, node):
        """Marks node as collected (updates in_degrees of successors and updates pending_nodes)."""
        self.pending_nodes.remove(node)
        for suc in self._direct_succs(node):
            self.in_degree[suc] -= 1
            if self.in_degree[suc] == 0:
                self.pending_nodes.append(suc)

    def collect_matching_block(self, filter_fn):
        """Iteratively collects the largest block of input (aka in_degree=0) nodes that match a
        given filtering function. Examples of this include collecting blocks of swap gates,
        blocks of linear gates (CXs and SWAPs), blocks of Clifford gates, blocks of single-qubit gates,
        blocks of two-qubit gates, etc..  Here 'iteratively' means that once a node is collected,
        the in_degrees of its immediate successors are decreased by 1, allowing more nodes to become
        leaf and to be eligible for collecting into the current block.
        Returns a block of collected nodes.
        """
        current_block = []
        unprocessed_pending_nodes = self.pending_nodes
        self.pending_nodes = []

        # Iteratively process unprocessed_pending_nodes:
        # - any node that cannot be collected is added to pending_nodes,
        # - any node that can be collected is added to the current_block,
        #   and some of its successors can be moved to unprocessed_pending_nodes.
        while unprocessed_pending_nodes:
            new_pending_nodes = []
            for node in unprocessed_pending_nodes:
                can_be_added = filter_fn(node)
                if can_be_added:
                    current_block.append(node)

                    # update the in_degree of node's successors
                    for suc in self._direct_succs(node):
                        self.in_degree[suc] -= 1
                        if self.in_degree[suc] == 0:
                            new_pending_nodes.append(suc)
                else:
                    self.pending_nodes.append(node)
            unprocessed_pending_nodes = new_pending_nodes

        return current_block

    def collect_commuting_block(self, filter_fn):
        """
        Collects the largest block of commuting input nodes that match a given
        filtering function. In a DagDependency nodes with in_degree=0 necessarily commute,
        and a node with in_degree>0 does not commute with any of its immediate predecessors.
        """
        current_block = []

        # nodes with in_degree=0 that match filter_fn
        for node in self.pending_nodes:
            if filter_fn(node):
                current_block.append(node)

        for node in current_block:
            self._mark_node_as_collected(node)

        return current_block

    def collect_parallel_block(self):
        """
        Collects the largest block of 'parallel' input nodes, where 'parallel' means that
        all nodes in the same block must be over pairwise disjoint qubits.
        """
        current_block = []
        current_qubits = set()

        # This is a simple greedy algorithm.
        # Though it seems beneficial to sort nodes by the number of qubits (from largest to smallest).
        self.pending_nodes.sort(key=lambda nd: -len(nd.qargs))

        for node in self.pending_nodes:
            node_qubits = set(node.qargs)
            # print(f"--{len(node_qubits)}")
            if current_qubits.isdisjoint(node_qubits):
                current_block.append(node)
                current_qubits = current_qubits.union(node_qubits)

        for node in current_block:
            self._mark_node_as_collected(node)

        return current_block

    def split_block_into_components(self, block):
        """Given a block of nodes, splits it into connected components."""

        if len(block) == 0:
            return block

        nodeset = set(block)
        colors = dict()
        color = -1

        def process_node_rec(node):
            """Given a node, recursively assigns the current color to all of its
            immediate successors and predecessors."""

            if node not in colors.keys():
                colors[node] = color

                for pred in self._direct_preds(node):
                    if node not in nodeset:
                        continue
                    process_node_rec(pred)

                for suc in self._direct_succs(node):
                    if node not in nodeset:
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

    def collect_all_matching_blocks(self, filter_fn):
        """Collects all blocks that match a given filtering function filter_fn.

        This iteratively finds the largest block that does not match filter_fn, then the largest block
        that matches filter_fn, and so on, until no more uncollected nodes remain.

        Intuitively, extracting larger blocks of non-matching nodes helps to find larger blocks of
        following matching nodes.

        For now, we return both the list of all blocks (including blocks of matching and non-matching
        nodes) and the list of matching blocks only.
        """

        def not_filter_fn(node):
            """Returns the opposite of filter_fn."""
            return not filter_fn(node)

        # Iteratively collect non-matching and matching blocks.
        matching_blocks = []
        all_blocks = []
        while self.have_uncollected_nodes():
            non_matching_block = self.collect_matching_block(not_filter_fn)
            if non_matching_block:
                all_blocks.append(non_matching_block)
            matching_block = self.collect_matching_block(filter_fn)
            if matching_block:
                all_blocks.append(matching_block)
                matching_blocks.append(matching_block)

        return matching_blocks, all_blocks

    def collect_all_commuting_blocks(self, filter_fn):
        """Collects all commuting blocks that match a given filtering function filter_fn.

        This iteratively finds the largest block that does not match filter_fn, then
        collects possibly multiple blocks of commuting nodes that match filter_fn,
        repeating the process until no more uncollected nodes remain.

        For now, we return both the list of all blocks and the list of commuting blocks.
        """

        def not_filter_fn(node):
            """Returns the opposite of filter_fn."""
            return not filter_fn(node)

        # Iteratively:
        #   Collect non-matching nodes.
        #   Collect layers of commuting nodes (while possible).
        matching_blocks = []
        all_blocks = []
        while self.have_uncollected_nodes():
            non_matching_block = self.collect_matching_block(not_filter_fn)
            if non_matching_block:
                all_blocks.append(non_matching_block)

            while True:
                commuting_block = self.collect_commuting_block(filter_fn)
                if commuting_block:
                    matching_blocks.append(commuting_block)
                    all_blocks.append(commuting_block)
                else:
                    break

        return matching_blocks, all_blocks

    def collect_all_parallel_blocks(self):
        """Collects all blocks of parallel nodes.

        Here, 'parallel'  means that nodes in the same block must have pairwise disjoint qubits.

        This is useful to try to rearrange nodes in the quantum circuit exploiting commutativity
        relations in the attempt to decrease the depth of the circuit.

        We return both the list of all blocks.
        """

        # Iteratively:
        #   Collect non-matching nodes.
        #   Collect layers of commuting nodes (while possible)
        all_blocks = []
        while self.have_uncollected_nodes():
            parallel_block = self.collect_parallel_block()
            assert parallel_block
            if parallel_block:
                all_blocks.append(parallel_block)
        return all_blocks
