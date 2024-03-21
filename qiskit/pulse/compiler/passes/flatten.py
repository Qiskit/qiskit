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

"""A flattening pass for Qiskit PulseIR compilation."""

from __future__ import annotations

from qiskit.pulse.compiler.basepasses import TransformationPass
from qiskit.pulse.ir import SequenceIR
from qiskit.pulse.exceptions import PulseCompilerError


class Flatten(TransformationPass):
    """Flatten ``SequenceIR`` object.

    The flattening process includes breaking up nested IRs until only instructions remain.
    After flattening the object will contain all instructions, timing information, and the
    complete sequence graph. However, the alignment of nested IRs will be lost. Because alignment
    information is essential for scheduling, flattening an unscheduled IR is not allowed.
    One should apply :class:`~qiskit.pulse.compiler.passes.SetSchedule` first.
    """

    def __init__(self):
        """Create new Flatten pass"""
        super().__init__(target=None)

    def run(
        self,
        passmanager_ir: SequenceIR,
    ) -> SequenceIR:
        """Run the pass."""

        self._flatten_recursive(passmanager_ir)
        return passmanager_ir

    # pylint: disable=cell-var-from-loop
    def _flatten_recursive(self, prog: SequenceIR) -> SequenceIR:
        """Recursively flatten the SequenceIR.

        Returns:
            A flattened ``SequenceIR`` object.

        Raises:
            PulseCompilerError: If ``prog`` is not scheduled.
        """
        # TODO : Consider replacing the alignment to "NullAlignment", as the original alignment
        #  has no meaning.

        def edge_map(_x, _y, _node):
            if _y == _node:
                return 0
            if _x == _node:
                return 1
            return None

        if any(prog.time_table[x] is None for x in prog.sequence.node_indices() if x not in (0, 1)):
            raise PulseCompilerError(
                "Can not flatten unscheduled IR. Use SetSchedule pass before Flatten."
            )

        for ind in prog.sequence.node_indices():
            if isinstance(sub_block := prog.sequence.get_node_data(ind), SequenceIR):
                self._flatten_recursive(sub_block)
                initial_time = prog.time_table[ind]
                nodes_mapping = prog.sequence.substitute_node_with_subgraph(
                    ind, sub_block.sequence, lambda x, y, _: edge_map(x, y, ind)
                )
                for old_node in nodes_mapping.keys():
                    if old_node not in (0, 1):
                        prog.time_table[nodes_mapping[old_node]] = (
                            initial_time + sub_block.time_table[old_node]
                        )

                del prog.time_table[ind]
                prog.sequence.remove_node_retain_edges(nodes_mapping[0])
                prog.sequence.remove_node_retain_edges(nodes_mapping[1])

        return prog

    def __hash__(self):
        return hash((self.__class__.__name__,))

    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__
