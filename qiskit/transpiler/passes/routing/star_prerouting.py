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
from typing import Union, List, Tuple
from math import floor, log10

from qiskit.circuit import Barrier
from qiskit.dagcircuit import DAGOpNode, DAGDepNode, DAGDependency, DAGCircuit
from qiskit.dagcircuit.collect_blocks import Block, BlockCollector
from qiskit.transpiler import Layout
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.library import SwapGate


class StarBlock(Block):
    """Defines blocks representing star-shaped pieces of a circuit."""

    def __init__(self, nodes=None, center=None, num2q=0):
        self.center = center
        self.num2q = num2q
        self.nodes = [] if nodes is None else nodes

    def append_node(self, node):
        """
        If node can be added to this block while keeping the block star-shaped,
        adds the node and returns ``True``. Otherwise, does not add the node
        and returns ``False``.
        """
        added = False

        if len(node.qargs) == 1:
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

    def size(self):
        """
        Returns the size of the block as measured by the number of
        two-qubits gates in this block.
        """
        return self.num2q

    def get_nodes(self):
        """Returns the list of nodes used in the block."""
        return self.nodes

    def split(self, split_blocks, split_layers):
        """
        In theory, this method can be used to split a star-shaped
        block into star-shaped sub-blocks. For now, this does not
        do anything and simply returns the list consisting of this
        very block.
        """
        return [self]

    def reverse(self):
        """
        Reversing a star-shaped block means reversing the nodes
        while keeping the same center.
        """
        return StarBlock(nodes=self.nodes[::-1], center=self.center, num2q=self.num2q)


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

    def __init__(self):
        """StarPreRouting"""
        super().__init__()

    def run(self, dag):
        # Extract StarBlocks from DAGCircuit / DAGDependency / DAGDependencyV2
        star_blocks, processing_order = self.determine_star_blocks_processing(dag, min_block_size=2)

        if not star_blocks:
            return dag

        if all(b.size() < 3 for b in star_blocks):
            # we only process blocks with less than 3 two-qubit gates in this pre-routing pass
            # if they occur in a collection of larger stars, otherwise we consider them to be 'lines'
            return dag

        # Create a new DAGCircuit / DAGDependency / DAGDependencyV2, replacing each
        # star block by a linear sequence of gates
        new_dag, qubit_mapping = self.star_preroute(dag, star_blocks, processing_order)

        # Fix output permutation -- copied from ElidePermutations
        input_qubit_mapping = {qubit: index for index, qubit in enumerate(dag.qubits)}
        self.property_set["original_layout"] = Layout(input_qubit_mapping)
        if self.property_set["original_qubit_indices"] is None:
            self.property_set["original_qubit_indices"] = input_qubit_mapping

        new_layout = Layout({dag.qubits[out]: idx for idx, out in enumerate(qubit_mapping)})
        if current_layout := self.property_set["virtual_permutation_layout"]:
            self.property_set["virtual_permutation_layout"] = new_layout.compose(
                current_layout.inverse(dag.qubits, dag.qubits), dag.qubits
            )
        else:
            self.property_set["virtual_permutation_layout"] = new_layout

        return new_dag

    def determine_star_blocks_processing(
        self, dag: Union[DAGCircuit, DAGDependency], min_block_size: int
    ) -> Tuple[List[StarBlock], Union[List[DAGOpNode], List[DAGDepNode]]]:
        """Returns star blocks in dag and the processing order of nodes within these star blocks
        Args:
            dag (DAGCircuit or DAGDependency): a dag on which star blocks should be determined.
            min_block_size (int): minimum number of two-qubit gates in a star block.

        Returns:
            List[StarBlock]: a list of star blocks in the given dag
            Union[List[DAGOpNode], List[DAGDepNode]]: a list of operations specifying processing order
        """

        def filter_fn(node):
            """Specifies which nodes can be collected into star blocks."""
            return (
                len(node.qargs) <= 2
                and len(node.cargs) == 0
                and getattr(node.op, "condition", None) is None
                and not isinstance(node.op, Barrier)
            )

        block_collector = BlockCollector(dag)
        matching_blocks = block_collector.collect_all_matching_blocks(
            filter_fn=filter_fn,
            block_class=StarBlock,
            output_nodes=False,
            min_block_size=min_block_size,
        )
        processing_order = [n for p in matching_blocks for n in p.nodes]

        return matching_blocks, processing_order

    def star_preroute(self, dag, blocks, processing_order):
        """Returns star blocks in dag and the processing order of nodes within these star blocks
        Args:
            dag (DAGCircuit or DAGDependency): a dag on which star prerouting should be performed.
            blocks (List[StarBlock]): a list of star blocks in the given dag.
            processing_order (Union[List[DAGOpNode], List[DAGDepNode]]): a list of operations specifying
            processing order

        Returns:
            new_dag: a dag specifying the pre-routed circuit
            qubit_mapping: the final qubit mapping after pre-routing
        """
        node_to_block_id = {}
        for i, block in enumerate(blocks):
            for node in block.get_nodes():
                node_to_block_id[node] = i

        new_dag = dag.copy_empty_like()
        processed_block_ids = set()
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

        int_digits = floor(log10(len(processing_order))) + 1
        processing_order_index_map = {
            node: f"a{str(index).zfill(int(int_digits))}"
            for index, node in enumerate(processing_order)
        }

        def tie_breaker_key(node):
            return processing_order_index_map.get(node, node.sort_key)

        for node in dag.topological_op_nodes(key=tie_breaker_key):
            block_id = node_to_block_id.get(node, None)
            if block_id is not None:
                if block_id in processed_block_ids:
                    continue

                processed_block_ids.add(block_id)

                # process the whole block
                block = blocks[block_id]
                sequence = block.nodes
                center_node = block.center

                if len(sequence) == 2:
                    for inner_node in sequence:
                        new_dag.apply_operation_back(
                            inner_node.op,
                            _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                            inner_node.cargs,
                            check=False,
                        )
                    continue
                swap_source = None
                prev = None
                for inner_node in sequence:
                    if (len(inner_node.qargs) == 1) or (inner_node.qargs == prev):
                        new_dag.apply_operation_back(
                            inner_node.op,
                            _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                            inner_node.cargs,
                            check=False,
                        )
                        continue
                    if is_first_star and swap_source is None:
                        swap_source = center_node
                        new_dag.apply_operation_back(
                            inner_node.op,
                            _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                            inner_node.cargs,
                            check=False,
                        )

                        prev = inner_node.qargs
                        continue
                    # place 2q-gate and subsequent swap gate
                    new_dag.apply_operation_back(
                        inner_node.op,
                        _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                        inner_node.cargs,
                        check=False,
                    )

                    if not inner_node is last_2q_gate and not isinstance(inner_node.op, Barrier):
                        new_dag.apply_operation_back(
                            SwapGate(),
                            _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                            inner_node.cargs,
                            check=False,
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
                    node.op,
                    _apply_mapping(node.qargs, qubit_mapping, dag.qubits),
                    node.cargs,
                    check=False,
                )
        return new_dag, qubit_mapping
