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
"""DAG representation of block components."""

from typing import Iterator, TYPE_CHECKING

import retworkx as rx

if TYPE_CHECKING:
    from .schedule import ScheduleComponent


class DAGSchedule:
    """DAG representation of :class:`~ScheduleBlock` components.

    This is topological ordering-aware block representation, which helps
    comparison of two schedule blocks without scheduling. For example,

    .. code-block::

        with pulse.build(default_alignment="left") as sched1:
            pulse.play(my_pulse1, pulse.DriveChannel(0))
            pulse.play(my_pulse2, pulse.DriveChannel(1))

        with pulse.build(default_alignment="left") as sched2:
            pulse.play(my_pulse2, pulse.DriveChannel(1))
            pulse.play(my_pulse1, pulse.DriveChannel(0))

        assert sched1 == sched2

    These schedules should be identical under the parallel alignment context,
    however the actual location of blocks in the timeslot is unknown before scheduling.
    In DAG representation, we can easily compare the topological odering of these components.
    """
    def __init__(self, is_sequential: bool):
        """Create new DAG.

        Args:
            is_sequential: Type of alignment context. Sequential context aligns
                nodes sequentially (set ``True``) while parallel context
                aligns nodes in different channels in parallel (set ``False``).
        """
        self._dag = rx.PyDAG()
        self._is_sequential = is_sequential

    def append(self, node: "ScheduleComponent"):
        """Add new node to DAG.

        Args:
            node: Schedule component to add.
        """
        if self._dag.num_nodes() == 0:
            self._dag.add_node(node)
        else:
            if self._is_sequential:
                prev_id = self._dag.node_indices()[-1]
                node_id = self._dag.add_node(node)
                edges = [(prev_id, node_id)]
            else:
                channels = set(node.channels)
                predecessors = []
                prev_ids = iter(reversed(self._dag.node_indices()))
                while len(channels) > 0:
                    try:
                        prev_id = next(prev_ids)
                    except StopIteration:
                        break
                    overlap = set(self._dag[prev_id].channels) & channels
                    if overlap:
                        predecessors.append(prev_id)
                        channels = channels - overlap
                node_id = self._dag.add_node(node)
                edges = [(prev_id, node_id) for prev_id in predecessors]
            self._dag.add_edges_from_no_data(edges)

    def replace(self, old: "ScheduleComponent", new: "ScheduleComponent"):
        """Replace old schedule component with new.

        Args:
            old: Old schedule component to be replaced.
            new: New schedule component to replace.
        """
        while True:
            node_id = self._dag.find_node_by_weight(old)
            if node_id is None:
                break
            self._dag[node_id] = new

    def nodes(self) -> Iterator["ScheduleComponent"]:
        """Return nodes.

        Yields:
            Nodes in the DAG.
        """
        yield from self._dag.nodes()

    def __eq__(self, other: "DAGSchedule"):
        return rx.is_isomorphic_node_match(self._dag, other._dag, lambda x, y: x == y)
