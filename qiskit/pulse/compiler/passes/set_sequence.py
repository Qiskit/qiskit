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

"""Sequencing pass for Qiskit PulseIR compilation."""

from __future__ import annotations
from functools import singledispatchmethod
from rustworkx import PyDAG

from qiskit.pulse.compiler.basepasses import TransformationPass
from qiskit.pulse.ir import SequenceIR
from qiskit.pulse.model import MixedFrame
from qiskit.pulse.transforms import AlignmentKind, SequentialAlignment, ParallelAlignment
from qiskit.pulse.exceptions import PulseCompilerError


class SetSequence(TransformationPass):
    """Sets the sequence of a :class:`.SequenceIR` object.

    The pass traverses the :class:`.SequenceIR` and recursively sets the sequence, by adding edges to
    the ``sequence`` property. Sequencing is done according to the alignment strategy.

    .. notes::
        The pass depends on the results of the analysis pass
        :class:`~qiskit.pulse.compiler.passes.MapMixedFrame`.
    """

    def __init__(self):
        """Create new SetSequence pass"""
        super().__init__(target=None)

    def run(
        self,
        passmanager_ir: SequenceIR,
    ) -> SequenceIR:
        """Run sequencing pass.

        Arguments:
            passmanager_ir: The IR object to be sequenced.

        Raises:
            PulseCompilerError: if ``property_set`` does not include a mixed_frames_mapping dictionary.
        """
        if self.property_set["mixed_frames_mapping"] is None:
            raise PulseCompilerError(
                "Parallel sequencing requires mixed frames mapping."
                " Run MapMixedFrame before sequencing"
            )
        self._sequence_instructions(passmanager_ir.alignment, passmanager_ir.sequence)
        return passmanager_ir

    @singledispatchmethod
    def _sequence_instructions(self, alignment: AlignmentKind, sequence: PyDAG):
        """Finalize the sequence by adding edges to the DAG

        ``sequence`` is mutated to include all the edges
        connecting the elements of the sequence.

        Nested :class:`.SequenceIR` objects are sequenced recursively.
        """
        raise NotImplementedError

    # pylint: disable=unused-argument
    @_sequence_instructions.register(ParallelAlignment)
    def _sequence_instructions_parallel(self, alignment: ParallelAlignment, sequence: PyDAG):
        """Finalize the sequence by adding edges to the DAG, following parallel alignment.

        ``sequence`` is mutated to include all the edges
        connecting the elements of the sequence in parallel.

        Nested :class:`.SequenceIR` objects are sequenced recursively.

        Args:
            alignment: The IR alignment.
            sequence: The graph object to be sequenced.
        """
        mixed_frame_mapping = self.property_set["mixed_frames_mapping"]

        idle_after = {}
        for ind in sequence.node_indices():
            if ind in (0, 1):
                # In, Out node
                continue
            node = sequence.get_node_data(ind)
            node_mixed_frames = set()

            if isinstance(node, SequenceIR):
                self._sequence_instructions(node.alignment, node.sequence)
                inst_targets = node.inst_targets
            else:
                inst_targets = [node.inst_target]

            for inst_target in inst_targets:
                if isinstance(inst_target, MixedFrame):
                    node_mixed_frames.add(inst_target)
                else:
                    node_mixed_frames |= mixed_frame_mapping[inst_target]

            pred_nodes = [
                idle_after[mixed_frame]
                for mixed_frame in node_mixed_frames
                if mixed_frame in idle_after
            ]
            if len(pred_nodes) == 0:
                pred_nodes = [0]
            for pred_node in pred_nodes:
                sequence.add_edge(pred_node, ind, None)
            for mixed_frame in node_mixed_frames:
                idle_after[mixed_frame] = ind
        sequence.add_edges_from_no_data([(ni, 1) for ni in idle_after.values()])

    # pylint: disable=unused-argument
    @_sequence_instructions.register(SequentialAlignment)
    def _sequence_instructions_sequential(self, alignment: SequentialAlignment, sequence: PyDAG):
        """Finalize the sequence by adding edges to the DAG, following sequential alignment.

        ``sequence`` is mutated to include all the edges
        connecting the elements of the sequence sequentially.

        Nested :class:`.SequenceIR` objects are sequenced recursively.

        Args:
            alignment: The IR alignment.
            sequence: The graph object to be sequenced.
        """
        nodes = sequence.node_indices()
        prev = 0
        # TODO : What's the correct order to use here? Addition index? Actual index?
        #  Should at least be documented.
        # The first two nodes are the in\out nodes.
        for ind in nodes[2:]:
            sequence.add_edge(prev, ind, None)
            prev = ind
            if isinstance(elem := sequence.get_node_data(ind), SequenceIR):
                self._sequence_instructions(elem.alignment, elem.sequence)
        sequence.add_edge(prev, 1, None)

    def __hash__(self):
        return hash((self.__class__.__name__,))

    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__
