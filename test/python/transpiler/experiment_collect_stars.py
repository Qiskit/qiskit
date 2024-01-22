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

from qiskit.circuit import QuantumCircuit
from qiskit.dagcircuit.collect_blocks import BlockCollector
from qiskit.converters import (
    circuit_to_dag,
    circuit_to_dagdependency,
    dag_to_circuit,
    dagdependency_to_circuit,
)


class StarBlock:

    def __init__(self, center=None, nodes=None):
        if nodes is None:
            nodes = []
        self.center = center
        self.nodes = [] if nodes is None else nodes

    def print(self):
        print(f"center: {self.center}, nodes: {self.nodes}")


def new_block_fn() -> StarBlock:
    return StarBlock()


def append_node_fn(node, block: StarBlock):
    """
    If node can be added to block while keeping the block star-shaped, adds node to block and
    return True. Otherwise, does not add node to block and returns False.
    """

    added = False

    print(f"append_node_fn: {node = }")
    block.print()

    if (
        len(node.qargs) == 2
        and len(node.cargs) == 0
        and getattr(node.op, "condition", None) is None
    ):

        if block.center is None:
            block.center = set(node.qargs)
            block.nodes.append(node)
            added = True

        elif isinstance(block.center, set):
            if node.qargs[0] in block.center:
                block.center = node.qargs[0]
                block.nodes.append(node)
                added = True
            elif node.qargs[1] in block.center:
                block.center = node.qargs[1]
                block.nodes.append(node)
                added = True

        else:
            if block.center in node.qargs:
                block.nodes.append(node)
                added = True

    return added


def get_nodes_fn(block: StarBlock):
    return block.nodes


def reverse_block_fn(block: StarBlock) -> StarBlock:
    return StarBlock(block.center, block.nodes[::-1])



if __name__ == "__main__":

    qc = QuantumCircuit(10)
    qc.h(0)
    qc.cx(0, range(1, 5))
    qc.h(9)
    qc.cx(9, range(8, 4, -1))
    print(qc)

    block_collector = BlockCollector(circuit_to_dag(qc))
    blocks = block_collector.collect_all_matching_blocks(
        new_block_fn=new_block_fn,
        append_node_fn=append_node_fn,
        get_nodes_fn=get_nodes_fn,
        reverse_block_fn=reverse_block_fn,
        split_block_fn=None,
    )

    print(f"Collected {len(blocks)} blocks:")
    for block in blocks:
        print(block)
