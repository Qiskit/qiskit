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

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.library import SwapGate, PermutationGate
from qiskit.dagcircuit.collect_blocks import BlockCollector, DefaultBlock


class StarBlock(DefaultBlock):
    def __init__(self, nodes=None, center=None, num2q=0):
        self.center = center
        self.num2q = num2q
        super().__init__(nodes)

    def print(self):
        print(f"==> center: {self.center}")
        print(f"==> num2q: {self.num2q}")

        for node in self.nodes:
            print(f"     {node.__repr__()}")

    def append_node(self, node):
        """
        If node can be added to block while keeping the block star-shaped, adds node to block and
        return True. Otherwise, does not add node to block and returns False.
        """

        added = False

        # ToDo: remove these assertions
        # These should be true because the filter_fn is supposed to accept only such nodes
        assert len(node.qargs) <= 2
        assert len(node.cargs) == 0
        assert getattr(node.op, "condition", None) is None

        if len(node.qargs) == 1:
            # This may be a bit sloppy since this may also include 1-qubit gates which are not part of the star.
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

    def run(self, dag):

        # Extract StarBlocks from DAGCircuit / DAGDependency / DAGDependencyV2
        block_collector = BlockCollector(dag)
        blocks = block_collector.collect_all_matching_blocks(
            filter_fn=filter_fn,
            split_layers=False,
            split_blocks=False,
            output_nodes=False,
            min_block_size=3,
            block_class=StarBlock,
        )

        if not blocks:
            return dag

        # Create a new DAGCircuit / DAGDependency / DAGDependencyV2, replacing each
        # star block by a linear sequence of gates
        node_to_block_id = {}
        for i, block in enumerate(blocks):
            for node in block.get_nodes():
                node_to_block_id[node] = i

        new_dag = dag.copy_empty_like()
        processed_block_ids = set()
        qubit_mapping = {bit: index for index, bit in enumerate(dag.qubits)}

        for node in dag.topological_op_nodes():
            block_id = node_to_block_id.get(node, None)
            if block_id is not None:
                if block_id not in processed_block_ids:
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
                            )
                            continue
                        if swap_source is None:
                            swap_source = center_node
                            new_dag.apply_operation_back(
                                inner_node.op,
                                _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                                inner_node.cargs,
                            )
                            prev = inner_node.qargs
                            continue
                        new_dag.apply_operation_back(
                            inner_node.op,
                            _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                            inner_node.cargs,
                        )
                        new_dag.apply_operation_back(
                            SwapGate(),
                            _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                            inner_node.cargs,
                        )
                        # Swap mapping
                        pos_0 = qubit_mapping[inner_node.qargs[0]]
                        pos_1 = qubit_mapping[inner_node.qargs[1]]
                        qubit_mapping[inner_node.qargs[0]] = pos_1
                        qubit_mapping[inner_node.qargs[1]] = pos_0
                        prev = inner_node.qargs

            else:
                # the node is not part of a block
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        pattern = [qubit_mapping[i] for i in dag.qubits]
        new_dag.apply_operation_back(PermutationGate(pattern), dag.qubits)

        return new_dag


def _apply_mapping(qargs, mapping, qubits):
    return tuple(qubits[mapping[x]] for x in qargs)
