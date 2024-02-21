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

"""
Temporary Compiler passes
"""
from __future__ import annotations
from collections import defaultdict
from functools import singledispatch
from rustworkx import PyDAG

from qiskit.pulse.model import MixedFrame
from qiskit.pulse.ir import IrBlock
from qiskit.pulse.ir.ir import InNode, OutNode
from qiskit.pulse.ir.alignments import (
    Alignment,
    AlignLeft,
    ParallelAlignment,
    AlignRight,
    SequentialAlignment,
    AlignSequential,
)


def analyze_target_frame_pass(ir_block: IrBlock, property_set) -> None:
    """Map the dependency of ``MixedFrame``s on ``PulseTarget`` and ``Frame``.

    Recursively traverses through the ``ir_block`` to find all ``MixedFrame``s,
    and maps their dependencies on ``PulseTarget`` and ``Frame``. The results
    are added as a dictionary to ``property_set`` under the key ``target_frame_map``.
    The added dictionary is keyed on every ``PulseTarget`` and ``Frame`` in ``ir_block``
    with the value being a set of all ``MixedFrame``s associated with the key.
    """
    target_frame_map = defaultdict(set)
    _analyze_target_frame_in_block(ir_block, target_frame_map)
    property_set["target_frame_map"] = dict(target_frame_map)


def _analyze_target_frame_in_block(ir_block: IrBlock, target_frame_map: dict) -> None:
    """A helper function to recurse through the block while mapping mixed frame dependency"""
    for elm in ir_block.elements():
        # Sub Block
        if isinstance(elm, IrBlock):
            _analyze_target_frame_in_block(elm, target_frame_map)
        # Pulse Instruction
        else:
            if isinstance(elm.inst_target, MixedFrame):
                target_frame_map[elm.inst_target.frame].add(elm.inst_target)
                target_frame_map[elm.inst_target.pulse_target].add(elm.inst_target)


def sequence_pass(ir_block: IrBlock, property_set: dict) -> IrBlock:
    """Finalize the sequence of the IrBlock by adding edges to the DAG"""
    _sequence_instructions(ir_block.alignment, ir_block._sequence, property_set)
    return ir_block


@singledispatch
def _sequence_instructions(alignment: Alignment, sequence: PyDAG, property_set: dict):
    """Finalize the sequence of the IrBlock by adding edges to the DAG"""
    raise NotImplementedError


@_sequence_instructions.register(ParallelAlignment)
def _sequence_parallel(alignment, sequence, property_set):
    """Sequence the IrBlock by recursively adding edges to the DAG,
    adding elements in parallel to one another in the graph"""
    idle_after = {}
    for ni in sequence.node_indices():
        if ni in (0, 1):
            # In, Out node
            continue
        node = sequence.get_node_data(ni)
        node_mixed_frames = set()

        if isinstance(node, IrBlock):
            # Recurse over sub block
            sequence_pass(node, property_set)
            inst_targets = node.inst_targets
        else:
            inst_targets = [node.inst_target]

        for inst_target in inst_targets:
            if isinstance(inst_target, MixedFrame):
                node_mixed_frames.add(inst_target)
            else:
                node_mixed_frames |= property_set["target_frame_map"][inst_target]

        pred_nodes = [
            idle_after[mixed_frame]
            for mixed_frame in node_mixed_frames
            if mixed_frame in idle_after
        ]
        if len(pred_nodes) == 0:
            pred_nodes = [0]
        for pred_node in pred_nodes:
            sequence.add_edge(pred_node, ni, None)
        for mixed_frame in node_mixed_frames:
            idle_after[mixed_frame] = ni
    sequence.add_edges_from_no_data([(ni, 1) for ni in idle_after.values()])


@_sequence_instructions.register(SequentialAlignment)
def _sequence_sequential(alignment, sequence: PyDAG, property_set):
    """Sequence the IrBlock by recursively adding edges to the DAG,
    adding elements one after the other"""
    sequence.add_edge(0, 2, None)
    sequence.add_edge(sequence.num_nodes() - 1, 1, None)
    sequence.add_edges_from_no_data([(x, x + 1) for x in range(2, sequence.num_nodes() - 1)])

    # Apply recursively
    for node in sequence.nodes():
        if isinstance(node, IrBlock):
            sequence_pass(node, property_set)


def schedule_pass(ir_block: IrBlock, property_set: dict) -> IrBlock:
    """Recursively schedule the block by setting initial time for every element"""
    # mutate graph
    _schedule_elements(ir_block.alignment, ir_block._time_table, ir_block._sequence, property_set)
    return ir_block


@singledispatch
def _schedule_elements(alignment, table, sequence, property_set):
    """Recursively schedule the block by setting initial time for every element"""
    raise NotImplementedError


@_schedule_elements.register(AlignLeft)
def _schedule_left_justified(alignment, table: dict, sequence: PyDAG, property_set: dict):
    """Recursively schedule the block by setting initial time for every element,
    aligning elements to the left."""

    first_nodes = sequence.successor_indices(0)
    nodes = []
    # Node 0 has no duration so is treated separately
    for node_index in first_nodes:
        table[node_index] = 0
        node = sequence.get_node_data(node_index)
        if isinstance(node, IrBlock):
            # Recruse over sub blocks
            schedule_pass(node, property_set)
        nodes.extend(sequence.successor_indices(node_index))

    while nodes:
        node_index = nodes.pop(0)
        if node_index == 1 or node_index in table:
            # reached end or already scheduled
            continue
        node = sequence.get_node_data(node_index)
        if isinstance(node, IrBlock):
            # Recruse over sub blocks
            schedule_pass(node, property_set)
        # Because we go in order, all predecessors are already scheduled
        preds = sequence.predecessor_indices(node_index)
        t0 = max([table.get(pred) + sequence.get_node_data(pred).duration for pred in preds])
        table[node_index] = t0
        nodes.extend(sequence.successor_indices(node_index))


@_schedule_elements.register(AlignSequential)
def _schedule_sequential(alignment, table: dict, sequence: PyDAG, property_set: dict):
    """Recursively schedule the block by setting initial time for every element,
    assuming the elements are sequential"""

    # TODO: This placeholder will fail if any change is done to the graph,
    #  needs a more robust implementation.

    total_time = 0

    for i in range(2, sequence.num_nodes()):
        table[i] = total_time
        node = sequence.get_node_data(i)
        if isinstance(node, IrBlock):
            schedule_pass(node, property_set)
        total_time += node.duration


@_schedule_elements.register(AlignRight)
def _schedule_right_justified(alignment, table: dict, sequence: PyDAG, property_set: dict) -> None:
    """Recursively schedule the block by setting initial time for every element,
    aligning elements to the right."""

    reversed_sequence = sequence.copy()
    # Reverse all edge
    reversed_sequence.reverse()
    # Now swap 0 and 1 nodes
    new_start_node_successors = reversed_sequence.successor_indices(1)
    new_end_node_predecessors = reversed_sequence.predecessor_indices(0)
    reversed_sequence.remove_node(0)
    reversed_sequence.remove_node(1)
    reversed_sequence.add_node(InNode)
    reversed_sequence.add_node(OutNode)
    reversed_sequence.add_edges_from_no_data([(0, x) for x in new_start_node_successors])
    reversed_sequence.add_edges_from_no_data([(x, 1) for x in new_end_node_predecessors])

    # Schedule with left alignment
    _schedule_left_justified(alignment, table, reversed_sequence, property_set)

    # Then reverse the timings
    total_duration = max(
        [
            table[i] + reversed_sequence.get_node_data(i).duration
            for i in reversed_sequence.predecessor_indices(1)
        ]
    )
    for key in table.keys():
        table[key] = total_duration - table[key] - reversed_sequence.get_node_data(key).duration
