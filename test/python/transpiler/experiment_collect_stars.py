# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Temporary code for experimental purposes."""

from qiskit.circuit import QuantumCircuit, Gate, Operation
from qiskit.circuit.library import SwapGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.collect_blocks import BlockCollector, DefaultBlock
from qiskit.converters import (
    circuit_to_dag,
    circuit_to_dagdependency,
    dag_to_circuit,
    dagdependency_to_circuit,
)


class StarBlock(DefaultBlock):

    def __init__(self, nodes=None, center=None):
        self.center = center
        super().__init__(nodes)

    def print(self):
        print(f"==> center: {self.center}")
        for node in self.nodes:
            print(f"     {node.__repr__()}")

    def append_node(self, node):
        """
        If node can be added to block while keeping the block star-shaped, adds node to block and
        return True. Otherwise, does not add node to block and returns False.
        """

        added = False

        assert len(node.qargs) <= 2
        assert len(node.cargs) == 0
        assert getattr(node.op, "condition", None) is None

        # print(f"In append:")
        # print(f"  => {node.op = }, {node.qargs = }")
        # block.print()

        if len(node.qargs) == 1:
            # This may be a bit sloppy since this may also include 1-qubit gates which are not part of the star.
            # Or maybe this is fine (need to think).
            self.nodes.append(node)
            added = True

        elif self.center is None:
            self.center = set(node.qargs)
            self.nodes.append(node)
            added = True

        elif isinstance(self.center, set):
            if node.qargs[0] in self.center:
                self.center = node.qargs[0]
                self.nodes.append(node)
                added = True
            elif node.qargs[1] in self.center:
                self.center = node.qargs[1]
                self.nodes.append(node)
                added = True

        else:
            if self.center in node.qargs:
                self.nodes.append(node)
                added = True

        # print(f" ==> {added = }")
        return added


    def reverse(self):
        return StarBlock(nodes=self.nodes, center=self.center)




def filter_fn(node):
    """Specifies which nodes can in principle be collected into 2-qubit blocks."""
    return len(node.qargs) <= 2 and len(node.cargs) == 0 and getattr(node.op, "condition", None) is None



class StarGate(Gate):

    def __init__(self, num_qubits, block):
        self.block = block
        print(f"STARGATE::init {num_qubits = }")
        super().__init__(name="star", num_qubits=num_qubits, params=[])






def resynthesize_stars_backup(dag):
    block_collector = BlockCollector(dag)
    blocks = block_collector.collect_all_matching_blocks(
        filter_fn=filter_fn,
        split_layers=False,
        split_blocks=False,
        output_nodes=False,
        block_class=StarBlock
    )

    print(f"Collected {len(blocks)} blocks:")
    for block in blocks:
        block.print()

    for block in blocks:
        num_qubits = len(dag.qubits)
        star = StarGate(num_qubits=num_qubits, block=block)
        wire_pos_map = {qb: ix for ix, qb in enumerate(dag.qubits)}
        dag.replace_block_with_op(block.get_nodes(), star, wire_pos_map, cycle_check=False)

    if isinstance(dag, DAGCircuit):
        qc = dag_to_circuit(dag)
        print(qc)

    else:
        qc = dagdependency_to_circuit(dag)
        print(qc)



def resynthesize_stars(dag):
    block_collector = BlockCollector(dag)
    blocks = block_collector.collect_all_matching_blocks(
        filter_fn=filter_fn,
        split_layers=False,
        split_blocks=False,
        output_nodes=False,
        block_class=StarBlock
    )

    print(f"Collected {len(blocks)} blocks:")
    for block in blocks:
        block.print()

    node_to_block_id = {}
    for i, block in enumerate(blocks):
        for node in block.get_nodes():
            node_to_block_id[node] = i

    print(f"{node_to_block_id = }")

    new_dag = dag.copy_empty_like()
    processed_block_ids = set()
    qubit_mapping = {bit: index for index, bit in enumerate(dag.qubits)}

    for topo_node in dag.topological_op_nodes():
        print(f"=> Considering {topo_node = }")

        block_id = node_to_block_id.get(topo_node, None)
        if block_id is not None:
            if block_id not in processed_block_ids:
                print(f" => new block id = {block_id}")
                processed_block_ids.add(block_id)

                # process the whole block
                block = blocks[block_id]
                sequence = block.nodes
                center_node = block.center

                print(f" => {sequence = }")
                print(f" => length = {len(sequence)}")
                print(f" => {center_node = }")

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
                    print(f" => {inner_node = }")
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
            # the node is not part of one of the blocks
            new_dag.apply_operation_back(topo_node.op, topo_node.qargs, topo_node.cargs)

    return new_dag


def _apply_mapping(qargs, mapping, qubits):
    return tuple(qubits[mapping[x]] for x in qargs)

# for block in blocks:
    #
    #     num_qubits = len(dag.qubits)
    #     star = StarGate(num_qubits=num_qubits, block=block)
    #     wire_pos_map = {qb: ix for ix, qb in enumerate(dag.qubits)}
    #     dag.replace_block_with_op(block.get_nodes(), star, wire_pos_map, cycle_check=False)
    #
    # if isinstance(dag, DAGCircuit):
    #     qc = dag_to_circuit(dag)
    #     print(qc)
    #
    # else:
    #     qc = dagdependency_to_circuit(dag)
    #     print(qc)


def example1():
    """Example similar to the one in Matthew's PR"""

    qc = QuantumCircuit(10)
    qc.h(0)
    qc.cx(0, range(1, 5))
    qc.h(9)
    qc.cx(9, range(8, 4, -1))
    print(qc)

    new_dag = resynthesize_stars(circuit_to_dag(qc))
    print_dag(new_dag)


def print_dag(dag):
    if isinstance(dag, DAGCircuit):
        qc = dag_to_circuit(dag)
        print(qc)

    else:
        qc = dagdependency_to_circuit(dag)
        print(qc)


def example2():
    """Showing off dag dependency"""

    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(3, 1)
    qc.cx(3, 2)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(3, 1)
    qc.cx(3, 2)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(3, 1)
    qc.cx(3, 2)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(3, 1)
    qc.cx(3, 2)
    print(qc)

    # print(f"First using DAGCircuit:")
    # new_dag = resynthesize_stars(circuit_to_dag(qc))
    # print_dag(new_dag)

    print(f"And now using DAGDependency:")
    new_dag = resynthesize_stars(circuit_to_dagdependency(qc))
    print_dag(new_dag)


if __name__ == "__main__":
    # example1()
    example2()
