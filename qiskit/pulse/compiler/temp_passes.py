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

from collections import defaultdict
from functools import singledispatch
from rustworkx import PyDAG

from qiskit.pulse.model import MixedFrame
from qiskit.pulse.ir import IrBlock
from qiskit.pulse.ir.alignments import Alignment, AlignLeft


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


@_sequence_instructions.register(AlignLeft)
def _sequence_left_justfied(alignment, sequence, property_set):
    """Finalize the sequence of the IrBlock by recursively adding edges to the DAG,
    aligning elements to the left."""
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
def _schedule_left_justfied(alignment, table, sequence: PyDAG, property_set):
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
