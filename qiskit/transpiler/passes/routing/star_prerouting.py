# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Search for star connectivity patterns and replace them with."""
from typing import Iterable

from qiskit.dagcircuit import DAGOpNode, DAGDepNode, DAGCircuitError, DAGCircuit, DAGDependency, \
    DAGDependencyV2
from qiskit.transpiler import Layout
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.library import SwapGate, PermutationGate
from qiskit.dagcircuit.collect_blocks import BlockCollector, DefaultBlock


class StarBlock(DefaultBlock):
    """Defines blocks representing star-shaped pieces of a circuit."""

    def __init__(self, nodes=None, center=None, num2q=0):
        self.center = center
        self.num2q = num2q
        super().__init__(nodes)

    def append_node(self, node):
        """
        If node can be added to block while keeping the block star-shaped, and
        return True. Otherwise, does not add node to block and returns False.
        """

        added = False

        # ToDo: remove these assertions
        # These should be true because the filter_fn is supposed to accept only such nodes
        assert len(node.qargs) <= 2
        assert len(node.cargs) == 0
        assert getattr(node.op, "condition", None) is None

        if len(node.qargs) == 1:
            # This may be a bit sloppy since this may also include 1-qubit gates
            # which are not part of the star.
            # Or maybe this is fine (need to think).
            self.nodes.append(node)
            added = True

        elif self.center is None:
            self.center = set(node.qargs)
            self.nodes.append(node)
            self.num2q += 1
            added = True

        elif isinstance(self.center, set):
            if node.qargs[0] in self.center:
                self.center = node.qargs[0]
                self.nodes.append(node)
                self.num2q += 1
                added = True
            elif node.qargs[1] in self.center:
                self.center = node.qargs[1]
                self.nodes.append(node)
                self.num2q += 1
                added = True

        else:
            if self.center in node.qargs:
                self.nodes.append(node)
                self.num2q += 1
                added = True

        return added

    def reverse(self):
        return StarBlock(nodes=self.nodes, center=self.center)

    def size(self):
        return self.num2q


def filter_fn(node):
    """Specifies which nodes can be collected into star blocks."""
    return (
        len(node.qargs) <= 2
        and len(node.cargs) == 0
        and getattr(node.op, "condition", None) is None
    )


class StarPreRouting(TransformationPass):
    """Run star to linear pre-routing

    This pass is a logical optimization pass that rewrites any
    solely 2q gate star connectivity subcircuit as a linear connectivity
    equivalent with swaps.

    For example:

      .. plot::
         :include-source:

         from qiskit.circuit import QuantumCircuit
         from qiskit.transpiler.passes import StarPreRouting

         qc = QuantumCircuit(10)
         qc.h(0)
         qc.cx(0, range(1, 5))
         qc.h(9)
         qc.cx(9, range(8, 4, -1))
         qc.measure_all()
         StarPreRouting()(qc).draw("mpl")

    This pass was inspired by a similar pass described in Section IV of:
    C. Campbell et al., "Superstaq: Deep Optimization of Quantum Programs,"
    2023 IEEE International Conference on Quantum Computing and Engineering (QCE),
    Bellevue, WA, USA, 2023, pp. 1020-1032, doi: 10.1109/QCE57702.2023.00116.
    """

    def __init__(
        self,
        add_permutation=False,
    ):
        """StarPreRouting"""

        self._pending_nodes: list[DAGOpNode | DAGDepNode] | None = None
        self._in_degree: dict[DAGOpNode | DAGDepNode, int] | None = None
        self._collect_from_back = False
        self.dag = None
        super().__init__()


    def _setup_in_degrees(self):
        """For an efficient implementation, for every node we keep the number of its
        unprocessed immediate predecessors (called ``_in_degree``). This ``_in_degree``
        is set up at the start and updated throughout the algorithm.
        A node is leaf (or input) node iff its ``_in_degree`` is 0.
        When a node is (marked as) collected, the ``_in_degree`` of each of its immediate
        successor is updated by subtracting 1.
        Additionally, ``_pending_nodes`` explicitly keeps the list of nodes whose
        ``_in_degree`` is 0.
        """
        self._pending_nodes = []
        self._in_degree = {}
        for node in self._op_nodes():
            deg = len(self._direct_preds(node))
            self._in_degree[node] = deg
            if deg == 0:
                self._pending_nodes.append(node)

    def _op_nodes(self) -> Iterable[DAGOpNode | DAGDepNode]:
        """Returns DAG nodes."""
        if not self.is_dag_dependency or self.is_v2:
            return self.dag.op_nodes()
        else:
            return self.dag.get_nodes()

    def _direct_preds(self, node):
        """Returns direct predecessors of a node. This function takes into account the
        direction of collecting blocks, that is node's predecessors when collecting
        backwards are the direct successors of a node in the DAG.
        """
        if not self.is_dag_dependency or self.is_v2:
            if self._collect_from_back:
                return [pred for pred in self.dag.successors(node) if isinstance(pred, DAGOpNode)]
            else:
                return [pred for pred in self.dag.predecessors(node) if isinstance(pred, DAGOpNode)]
        else:
            if self._collect_from_back:
                return [
                    self.dag.get_node(pred_id)
                    for pred_id in self.dag.direct_successors(node.node_id)
                ]
            else:
                return [
                    self.dag.get_node(pred_id)
                    for pred_id in self.dag.direct_predecessors(node.node_id)
                ]

    def _direct_succs(self, node):
        """Returns direct successors of a node. This function takes into account the
        direction of collecting blocks, that is node's successors when collecting
        backwards are the direct predecessors of a node in the DAG.
        """
        if not self.is_dag_dependency or self.is_v2:
            if self._collect_from_back:
                return [succ for succ in self.dag.predecessors(node) if isinstance(succ, DAGOpNode)]
            else:
                return [succ for succ in self.dag.successors(node) if isinstance(succ, DAGOpNode)]
        else:
            if self._collect_from_back:
                return [
                    self.dag.get_node(succ_id)
                    for succ_id in self.dag.direct_predecessors(node.node_id)
                ]
            else:
                return [
                    self.dag.get_node(succ_id)
                    for succ_id in self.dag.direct_successors(node.node_id)
                ]

    def _have_uncollected_nodes(self):
        """Returns whether there are uncollected (pending) nodes"""
        return len(self._pending_nodes) > 0

    def collect_matching_block(self, filter_fn, block_class=DefaultBlock, output_nodes=True):
        """Iteratively collects the largest block of input nodes (that is, nodes with
        ``_in_degree`` equal to 0) that match a given filtering function.
        Examples of this include collecting blocks of swap gates,
        blocks of linear gates (CXs and SWAPs), blocks of Clifford gates, blocks of single-qubit gates,
        blocks of two-qubit gates, etc.  Here 'iteratively' means that once a node is collected,
        the ``_in_degree`` of each of its immediate successor is decreased by 1, allowing more nodes
        to become input and to be eligible for collecting into the current block.
        Returns the block of collected nodes.
        """
        unprocessed_pending_nodes = self._pending_nodes
        self._pending_nodes = []

        current_block = block_class()

        # Iteratively process unprocessed_pending_nodes:
        # - any node that does not match filter_fn is added to pending_nodes
        # - any node that match filter_fn is added to the current_block,
        #   and some of its successors may be moved to unprocessed_pending_nodes.
        while unprocessed_pending_nodes:
            new_pending_nodes = []
            for node in unprocessed_pending_nodes:
                added = filter_fn(node) and current_block.append_node(node)
                if added:
                    # update the _in_degree of node's successors
                    for suc in self._direct_succs(node):
                        self._in_degree[suc] -= 1
                        if self._in_degree[suc] == 0:
                            new_pending_nodes.append(suc)
                else:
                    self._pending_nodes.append(node)
            unprocessed_pending_nodes = new_pending_nodes

        return current_block.get_nodes() if output_nodes else current_block

    def collect_all_matching_blocks(
        self,
        filter_fn,
        split_blocks=True,
        min_block_size=2,
        split_layers=False,
        collect_from_back=False,
        block_class=DefaultBlock,
        output_nodes=True,
    ):
        """Collects all blocks that match a given filtering function filter_fn.
        This iteratively finds the largest block that does not match filter_fn,
        then the largest block that matches filter_fn, and so on, until no more uncollected
        nodes remain. Intuitively, finding larger blocks of non-matching nodes helps to
        find larger blocks of matching nodes later on.

        After the blocks are collected, they can be optionally refined. The option
        ``split_blocks`` allows to split collected blocks into sub-blocks over disjoint
        qubit subsets. The option ``split_layers`` allows to split collected blocks
        into layers of non-overlapping instructions. The option ``min_block_size``
        specifies the minimum number of gates in the block for the block to be collected.

        By default, blocks are collected in the direction from the inputs towards the outputs
        of the circuit. The option ``collect_from_back`` allows to change this direction,
        that is collect blocks from the outputs towards the inputs of the circuit.

        Returns the list of matching blocks only.
        """

        def not_filter_fn(node):
            """Returns the opposite of filter_fn."""
            return not filter_fn(node)

        # Note: the collection direction must be specified before setting in-degrees
        self._collect_from_back = collect_from_back
        self._setup_in_degrees()

        # Iteratively collect non-matching and matching blocks.
        matching_blocks: list[list[DAGOpNode | DAGDepNode]] = []
        processing_order = []
        while self._have_uncollected_nodes():
            self.collect_matching_block(
                filter_fn=not_filter_fn,
                output_nodes=True,
            )
            matching_block = self.collect_matching_block(
                filter_fn=filter_fn, block_class=block_class, output_nodes=False
            )
            if matching_block.size() >= min_block_size:
                matching_blocks.append(matching_block)
            processing_order.append(matching_block)
        if split_blocks or split_layers:
            tmp_blocks = []
            for block in matching_blocks:
                tmp_blocks.extend(block.split(split_blocks=split_blocks, split_layers=split_layers))
            matching_blocks = tmp_blocks

        # If we are collecting from the back, both the order of the blocks
        # and the order of nodes in each block should be reversed.
        if self._collect_from_back:
            matching_blocks = [block.reverse() for block in matching_blocks[::-1]]

        # Keep only blocks with at least min_block_sizes.
        matching_blocks = [
            block for block in matching_blocks if len(block.get_nodes()) >= min_block_size
        ]
        processing_order = [n for p in processing_order for n in p.nodes]
        if output_nodes:
            return [block.get_nodes() for block in matching_blocks], processing_order

        return matching_blocks, processing_order


    def run(self, dag):
        self.dag = dag
        if isinstance(dag, DAGCircuit):
            self.is_dag_dependency = False
            self.is_v2 = False

        elif isinstance(dag, DAGDependency):
            self.is_dag_dependency = True
            self.is_v2 = False

        elif isinstance(dag, DAGDependencyV2):
            self.is_dag_dependency = True
            self.is_v2 = True

        else:
            raise DAGCircuitError("not a DAG.")
        # Extract StarBlocks from DAGCircuit / DAGDependency / DAGDependencyV2

        self.blocks, processing_order = self.collect_all_matching_blocks(
            filter_fn=filter_fn,
            split_layers=False,
            split_blocks=False,
            output_nodes=False,
            min_block_size=2,
            block_class=StarBlock,
        )

        if not self.blocks:
            return dag

        # Create a new DAGCircuit / DAGDependency / DAGDependencyV2, replacing each
        # star block by a linear sequence of gates
        node_to_block_id = {}
        for i, block in enumerate(self.blocks):
            for node in block.get_nodes():
                node_to_block_id[node] = i

        new_dag = dag.copy_empty_like()
        processed_block_ids = set()
        # qubit_mapping = {bit: index for index, bit in enumerate(dag.qubits)}
        qubit_mapping = list(range(len(dag.qubits)))

        def _apply_mapping(qargs, qubit_mapping, qubits):
            return tuple(qubits[qubit_mapping[dag.find_bit(qubit).index]] for qubit in qargs)

        is_first_star = True
        last_2q_gate = [
            op
            for op in reversed(processing_order)
            if ((len(op.qargs) > 1) and (op.name != "barrier"))
        ]
        if len(last_2q_gate) > 0:
            last_2q_gate = last_2q_gate[0]
        else:
            last_2q_gate = None

        is_processed = {p: False for p in processing_order}
        for node in processing_order:
            block_id = node_to_block_id.get(node, None)
            if block_id is not None:
                if block_id in processed_block_ids or is_processed[node]:
                    continue

                processed_block_ids.add(block_id)
                is_processed[node] = True
                # process the whole block
                block = self.blocks[block_id]
                sequence = block.nodes
                center_node = block.center

                if len(sequence) == 2:
                    for inner_node in sequence:
                        new_dag.apply_operation_back(
                            inner_node.op,
                            _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                            inner_node.cargs,
                        )
                        is_processed[inner_node] = True
                    continue
                swap_source = None
                prev = None
                for inner_node in sequence:
                    if (len(inner_node.qargs) == 1) or (inner_node.qargs == prev):
                        if inner_node is sequence[-1]:
                            print()
                        new_dag.apply_operation_back(
                            inner_node.op,
                            _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                            inner_node.cargs,
                        )
                        is_processed[inner_node] = True
                        continue
                    if is_first_star and swap_source is None:
                        swap_source = center_node
                        new_dag.apply_operation_back(
                            inner_node.op,
                            _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                            inner_node.cargs,
                        )
                        is_processed[inner_node] = True

                        prev = inner_node.qargs
                        continue
                    # place 2q-gate and subsequent swap gate
                    new_dag.apply_operation_back(
                        inner_node.op,
                        _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                        inner_node.cargs,
                    )
                    is_processed[inner_node] = True

                    if not inner_node is last_2q_gate:
                        new_dag.apply_operation_back(
                            SwapGate(),
                            _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                            inner_node.cargs,
                        )
                        # Swap mapping
                        index_0 = dag.find_bit(inner_node.qargs[0]).index
                        index_1 = dag.find_bit(inner_node.qargs[1]).index
                        qubit_mapping[index_1], qubit_mapping[index_0] = (
                            qubit_mapping[index_0],
                            qubit_mapping[index_1],
                        )

                    prev = inner_node.qargs
                is_first_star = False
            else:
                # the node is not part of a block
                new_dag.apply_operation_back(
                    node.op, _apply_mapping(node.qargs, qubit_mapping, dag.qubits), node.cargs
                )
        """
        if self.add_permutation:
            pattern = [qubit_mapping[i] for i in dag.qubits]
            new_dag.apply_operation_back(PermutationGate(pattern), dag.qubits)
        """

        # copied from ElidePermutations
        input_qubit_mapping = {qubit: index for index, qubit in enumerate(dag.qubits)}
        self.property_set["original_layout"] = Layout(input_qubit_mapping)
        if self.property_set["original_qubit_indices"] is None:
            self.property_set["original_qubit_indices"] = input_qubit_mapping
        # ToDo: check if this exists; then compose
        self.property_set["virtual_permutation_layout"] = Layout(
            {dag.qubits[out]: idx for idx, out in enumerate(qubit_mapping)}
        )

        self.property_set["stars"] = self.blocks
        self.property_set["star_dag"] = new_dag
        return new_dag



"""
def _apply_mapping(qargs, mapping, qubits):
    return tuple(qubits[mapping[x]] for x in qargs)
"""
