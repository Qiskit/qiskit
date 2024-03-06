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

# pylint: disable=cyclic-import

"""
=========
Pulse IR
=========
"""

from __future__ import annotations
from collections import defaultdict

import copy

import rustworkx as rx
from rustworkx import PyDAG
from rustworkx.visualization import graphviz_draw

from qiskit.pulse.ir.alignments import Alignment
from qiskit.pulse import Instruction
from qiskit.pulse.exceptions import PulseError

InNode = object()
OutNode = object()


class SequenceIR:
    """IR representation of instruction sequences

    ``SequenceIR`` is the backbone of the intermediate representation used in the Qiskit Pulse compiler.
    A pulse program is represented as a single ``SequenceIR`` object, with elements
    which include ``IrInstruction`` objects and other nested ``SequenceIR`` objects.
    """

    def __init__(self, alignment: Alignment):
        self._alignment = alignment

        self._time_table = defaultdict(lambda: None)
        self._sequence = rx.PyDAG(multigraph=False)
        self._sequence.add_nodes_from([InNode, OutNode])

    @property
    def alignment(self) -> Alignment:
        """Return the alignment of the SequenceIR"""
        return self._alignment

    @property
    def inst_targets(self) -> set:
        """Recursively return a set of all Instruction.inst_target in the SequenceIR"""
        inst_targets = set()
        for elm in self.elements():
            if isinstance(elm, SequenceIR):
                inst_targets |= elm.inst_targets
            else:
                inst_targets.add(elm.inst_target)
        return inst_targets

    @property
    def sequence(self) -> PyDAG:
        """Return the DAG sequence of the SequenceIR"""
        return self._sequence

    def append(self, element: SequenceIR | Instruction) -> int:
        """Append element to the SequenceIR

        Returns: The index of the added element in the sequence.
        """
        return self._sequence.add_node(element)

    def elements(self) -> list[SequenceIR | Instruction]:
        """Return a list of all elements in the SequenceIR"""
        return self._sequence.nodes()[2:]

    def scheduled_elements(
        self, recursive: bool = False
    ) -> list[tuple[int | None, SequenceIR | Instruction]]:
        """Return a list of scheduled elements.

        Each element in the list is [initial_time, element].
        """
        if recursive:
            return self._recursive_scheduled_elements(0)
        else:
            return [
                (self._time_table[ind], self._sequence.get_node_data(ind))
                for ind in self._sequence.node_indices()
                if ind not in (0, 1)
            ]

    def _recursive_scheduled_elements(
        self, time_offset: int
    ) -> list[tuple[int | None, SequenceIR | Instruction]]:
        """Helper function to recursively return the scheduled elements.

        The absolute timing is tracked via the `time_offset`` argument, which represents the initial time
        of the block itself.

        Each element in the list is a tuple (initial_time, element).
        """
        scheduled_elements = []
        for ind in self._sequence.node_indices():
            if ind not in [0, 1]:
                node = self._sequence.get_node_data(ind)
                try:
                    time = self._time_table[ind] + time_offset
                except TypeError as ex:
                    raise PulseError(
                        "Can not return recursive list of scheduled elements"
                        " if sub blocks are not scheduled."
                    ) from ex

                if isinstance(node, SequenceIR):
                    scheduled_elements.extend(node._recursive_scheduled_elements(time))
                else:
                    scheduled_elements.append((time, node))

        return scheduled_elements

    def initial_time(self) -> int | None:
        """Return initial time"""
        first_nodes = self._sequence.successor_indices(0)
        if not first_nodes:
            return None
        return min([self._time_table[ind] for ind in first_nodes], default=None)

    def final_time(self) -> int | None:
        """Return final time"""
        last_nodes = self._sequence.predecessor_indices(1)
        if not last_nodes:
            return None
        tf = None
        for ind in last_nodes:
            if (t0 := self._time_table[ind]) is not None:
                duration = self._sequence.get_node_data(ind).duration
                if tf is None:
                    tf = t0 + duration
                else:
                    tf = max(tf, t0 + duration)
        return tf

    @property
    def duration(self) -> int | None:
        """Return the duration of the SequenceIR"""
        try:
            return self.final_time() - self.initial_time()
        except TypeError:
            return None

    def draw(self, recursive: bool = False):
        """Draw the graph of the SequenceIR"""
        if recursive:
            draw_sequence = self.flatten().sequence
        else:
            draw_sequence = self.sequence

        def _draw_nodes(n):
            if n is InNode or n is OutNode:
                return {"fillcolor": "grey", "style": "filled"}
            try:
                name = " " + n.name
            except (TypeError, AttributeError):
                name = ""
            return {"label": f"{n.__class__.__name__}" + name}

        return graphviz_draw(
            draw_sequence,
            node_attr_fn=_draw_nodes,
        )

    def flatten(self, inplace: bool = False) -> SequenceIR:
        """Recursively flatten the SequenceIR"""
        # TODO : Verify that the block\sub blocks are sequenced correctly.
        if inplace:
            block = self
        else:
            block = copy.deepcopy(self)
            block._sequence[0] = InNode
            block._sequence[1] = OutNode
            # TODO : Move this to __deepcopy__

        def edge_map(x, y, node):
            if y == node:
                return 0
            if x == node:
                return 1
            return None

        for ind in block.sequence.node_indices():
            if isinstance(sub_block := block.sequence.get_node_data(ind), SequenceIR):
                sub_block.flatten(inplace=True)
                initial_time = block._time_table[ind]
                nodes_mapping = block._sequence.substitute_node_with_subgraph(
                    ind, sub_block.sequence, lambda x, y, _: edge_map(x, y, ind)
                )
                if initial_time is not None:
                    for old_node in nodes_mapping.keys():
                        if old_node not in (0, 1):
                            block._time_table[nodes_mapping[old_node]] = (
                                initial_time + sub_block._time_table[old_node]
                            )
                block._sequence.remove_node_retain_edges(nodes_mapping[0])
                block._sequence.remove_node_retain_edges(nodes_mapping[1])

        return block

    def __eq__(self, other: SequenceIR):
        if not isinstance(other.alignment, type(self.alignment)):
            return False
        # TODO : This is a temporary setup until we figure out the
        #  alignment hierarchy and set equating there.
        return rx.is_isomorphic_node_match(self._sequence, other._sequence, lambda x, y: x == y)

        # TODO : What about the time_table? The isomorphic comparison allows for the indices
        #  to be different, But then it's not straightforward to compare the time_table.
        #  It is reasonable to assume that blocks with the same alignment and the same sequence
        #  will result in the same time_table, but this decision should be made consciously.
