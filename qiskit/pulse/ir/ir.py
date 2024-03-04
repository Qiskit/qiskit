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

import copy

import rustworkx as rx
from rustworkx import PyDAG
from rustworkx.visualization import graphviz_draw

from qiskit.pulse.ir.alignments import Alignment
from qiskit.pulse import Instruction

InNode = object()
OutNode = object()


class IrBlock:
    """IR representation of instruction sequences

    ``IrBlock`` is the backbone of the intermediate representation used in the Qiskit Pulse compiler.
    A pulse program is represented as a single ``IrBlock`` object, with elements
    which include ``IrInstruction`` objects and other nested ``IrBlock`` objects.
    """

    def __init__(self, alignment: Alignment):
        self._alignment = alignment

        self.time_offset = 0
        self._time_table = {}
        self._children = []
        self._sequence = rx.PyDAG(multigraph=False)
        self._sequence.add_nodes_from([InNode, OutNode])

    @property
    def alignment(self) -> Alignment:
        """Return the alignment of the IrBlock"""
        return self._alignment

    @property
    def inst_targets(self) -> set:
        """Recursively return a set of all Instruction.inst_target in the IrBlock"""
        inst_targets = set()
        for elm in self.elements():
            if isinstance(elm, IrBlock):
                inst_targets |= elm.inst_targets
            else:
                inst_targets.add(elm.inst_target)
        return inst_targets

    @property
    def sequence(self) -> PyDAG:
        """Return the DAG sequence of the IrBlock"""
        return self._sequence

    def append(self, element: IrBlock | Instruction) -> None:
        """Append element to the IrBlock"""
        new_node_id = self._sequence.add_node(element)
        if isinstance(element, IrBlock):
            self._children.append(new_node_id)

    def elements(self) -> list[IrBlock | Instruction]:
        """Return a list of all elements in the IrBlock"""
        return self._sequence.nodes()[2:]

    def scheduled_elements(self) -> list[list[int | None, IrBlock | Instruction]]:
        """Return a list of scheduled elements.

        Each element in the list is [initial_time, element].
        """
        return [
            [self._time_table.get(ni, None), self._sequence.get_node_data(ni)]
            for ni in self._sequence.node_indices()
            if ni not in (0, 1)
        ]

    def initial_time(self) -> int | None:
        """Return initial time"""
        first_nodes = self._sequence.successor_indices(0)
        if not first_nodes:
            return None
        return min([self._time_table.get(node, None) for node in first_nodes], default=None)

    def final_time(self) -> int | None:
        """Return final time"""
        last_nodes = self._sequence.predecessor_indices(1)
        if not last_nodes:
            return None
        tf = None
        for ni in last_nodes:
            if (t0 := self._time_table.get(ni, None)) is not None:
                duration = self._sequence.get_node_data(ni).duration
                if tf is None:
                    tf = t0 + duration
                else:
                    tf = max(tf, t0 + duration)
        return tf

    @property
    def duration(self) -> int | None:
        """Return the duration of the IrBlock"""
        try:
            return self.final_time() - self.initial_time()
        except TypeError:
            return None

    def draw(self, recursive: bool = False):
        """Draw the graph of the IrBlock"""
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

    def flatten(self, inplace: bool = False) -> IrBlock:
        """Recursively flatten the IrBlock"""
        # TODO : Verify that the block\sub blocks are sequenced.
        if inplace:
            block = self
        else:
            block = copy.deepcopy(self)
            block._sequence[0] = InNode
            block._sequence[1] = OutNode

        for ind in block.sequence.node_indices():
            if isinstance(block.sequence.get_node_data(ind), IrBlock):

                def edge_map(x, y, w):
                    if y == ind:
                        return 0
                    if x == ind:
                        return 1
                    return None

                sub_block = block.sequence.get_node_data(ind)
                sub_block.flatten(inplace=True)
                initial_time = block._time_table.get(ind, None)
                nodes_mapping = block._sequence.substitute_node_with_subgraph(
                    ind, sub_block.sequence, edge_map
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

    def __eq__(self, other):
        if self.alignment != other.alignment:
            return False
        # TODO : Define alignment equating. Currently this doesn't work.
        return rx.is_isomorphic_node_match(
            self._sequence, other._sequence, lambda x, y: x == y or x is y
        )
