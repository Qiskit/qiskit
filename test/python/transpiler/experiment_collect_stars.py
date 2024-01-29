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






def example1():
    """Example similar to the one in Matthew's PR"""

    qc = QuantumCircuit(10)
    qc.h(0)
    qc.cx(0, range(1, 5))
    qc.h(9)
    qc.cx(9, range(8, 4, -1))
    print(qc)

    block_collector = BlockCollector(circuit_to_dag(qc))
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

    print(f"First using DAGCircuit:")


    block_collector = BlockCollector(circuit_to_dag(qc))
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

    print(f"And now using DAGDependency:")

    block_collector = BlockCollector(circuit_to_dagdependency(qc))
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


if __name__ == "__main__":
    example1()
    example2()
