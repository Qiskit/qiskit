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
        add_permutation=True,
    ):
        """StarPreRouting"""
        self.add_permutation = add_permutation
        self.blocks = None
        super().__init__()

    def run(self, dag):

        # Extract StarBlocks from DAGCircuit / DAGDependency / DAGDependencyV2
        block_collector = BlockCollector(dag)
        self.blocks = block_collector.collect_all_matching_blocks(filter_fn=filter_fn,
                                                                  split_layers=False, split_blocks=False,
                                                                  output_nodes=False, min_block_size=2,
                                                                  block_class=StarBlock, )

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
        #qubit_mapping = {bit: index for index, bit in enumerate(dag.qubits)}
        qubit_mapping = list(range(len(dag.qubits)))

        def _apply_mapping(qargs, qubit_mapping, qubits):
            return tuple(qubits[qubit_mapping[dag.find_bit(qubit).index]] for qubit in qargs)

        is_first_star = True
        topological_nodes = list(dag.topological_op_nodes())
        last_2q_gate = [op for op in reversed(topological_nodes) if ((len(op.qargs) > 1) and (op.name != "barrier"))]
        if len(last_2q_gate) > 0:
            last_2q_gate = last_2q_gate[0]
        else:
            last_2q_gate = None

        for node in topological_nodes:
            block_id = node_to_block_id.get(node, None)
            if block_id is not None:
                if block_id not in processed_block_ids:
                    processed_block_ids.add(block_id)
                    is_last_block = len(self.blocks) == len(processed_block_ids)
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
                        if is_first_star and swap_source is None:
                            swap_source = center_node
                            new_dag.apply_operation_back(
                                inner_node.op,
                                _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                                inner_node.cargs,
                            )
                            prev = inner_node.qargs
                            continue
                        # place 2q-gate and subsequent swap gate
                        new_dag.apply_operation_back(
                            inner_node.op,
                            _apply_mapping(inner_node.qargs, qubit_mapping, dag.qubits),
                            inner_node.cargs,
                        )
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
                new_dag.apply_operation_back(node.op, _apply_mapping(node.qargs, qubit_mapping, dag.qubits), node.cargs)
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
        self.property_set['star_dag'] = new_dag
        return new_dag




"""
def _apply_mapping(qargs, mapping, qubits):
    return tuple(qubits[mapping[x]] for x in qargs)
"""
