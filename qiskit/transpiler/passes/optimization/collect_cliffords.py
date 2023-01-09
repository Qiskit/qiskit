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


"""Replace each sequence of Clifford gates by a single Clifford gate."""

from functools import partial

from qiskit.transpiler.passes.optimization.collect_and_collapse import (
    CollectAndCollapse,
    collect_using_filter_function,
    collapse_to_operation,
)

from qiskit.quantum_info.operators import Clifford


class CollectCliffords(CollectAndCollapse):
    """Collects blocks of Clifford gates and replaces them by a :class:`~qiskit.quantum_info.Clifford`
    object.
    """

    def __init__(self, do_commutative_analysis=False, split_blocks=True, min_block_size=2):
        """CollectCliffords initializer.

        Args:
            do_commutative_analysis (bool): if True, exploits commutativity relations
                between nodes.
            split_blocks (bool): if True, splits collected blocks into sub-blocks
                over disjoint qubit subsets.
            min_block_size (int): specifies the minimum number of gates in the block
                for the block to be collected.
        """

        collect_function = partial(
            collect_using_filter_function,
            filter_function=_is_clifford_gate,
            split_blocks=split_blocks,
            min_block_size=min_block_size,
        )
        collapse_function = partial(collapse_to_operation, collapse_function=_collapse_to_clifford)

        super().__init__(
            collect_function=collect_function,
            collapse_function=collapse_function,
            do_commutative_analysis=do_commutative_analysis,
        )


clifford_gate_names = ["x", "y", "z", "h", "s", "sdg", "cx", "cy", "cz", "swap"]


def _is_clifford_gate(node):
    """Specifies whether a node holds a clifford gate."""
    return node.op.name in clifford_gate_names and getattr(node.op, "condition", None) is None


def _collapse_to_clifford(circuit):
    """Specifies how to construct a ``Clifford`` from a quantum circuit (that must
    consist of Clifford gates only)."""
    return Clifford(circuit)
