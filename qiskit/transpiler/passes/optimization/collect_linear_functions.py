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


"""Replace each sequence of CX and SWAP gates by a single LinearFunction gate."""

from qiskit.circuit.library.generalized_gates import LinearFunction
from qiskit.dagcircuit.collect_blocks import BlockCollector, BlockSplitter
from qiskit.transpiler.passes.optimization.collapse_chains import CollapseChains
from qiskit.circuit import QuantumCircuit, Operation


class CollectLinearFunctions(CollapseChains):
    """Collect blocks of linear gates (:class:`.CXGate` and :class:`.SwapGate` gates)
    and replaces them by linear functions (:class:`.LinearFunction`)."""

    def filter_function(self, node):
        """Specifies which nodes to collect into blocks."""
        return node.op.name in ("cx", "swap") and getattr(node.op, "condition", None) is None

    def collapse_function(
        self,
        circuit: QuantumCircuit,
    ) -> Operation:
        """Specifies what to do with the collected blocks.

        Args:
            circuit (QuantumCircuit): quantum circuit containing the block of nodes
                to be collapsed (combined).

        Returns:
            Operation: the result of combining nodes in ``circuit``.
        """
        return LinearFunction(circuit)


# The functions below are no longer needed and should possibly be removed/deprecated.


def collect_linear_blocks(dag):
    """Collect blocks of linear gates."""

    def is_linear(node):
        return node.op.name in ["cx", "swap"] and getattr(node.op, "condition", None) is None

    blocks = BlockCollector(dag).collect_all_matching_blocks(filter_fn=is_linear)
    return blocks


def split_block_into_components(dag, block):
    """Given a block of gates, splits it into connected components."""
    return BlockSplitter().run(block)


def split_blocks_into_components(dag, blocks):
    """Given blocks of gates, splits each block into connected components,
    and returns a list of all blocks."""
    split_blocks = []
    for block in blocks:
        split_blocks.extend(split_block_into_components(dag, block))
    return split_blocks
