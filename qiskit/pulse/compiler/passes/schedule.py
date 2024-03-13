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

"""A scheduling pass for Qiskit PulseIR compilation."""

from __future__ import annotations
from functools import singledispatchmethod
from collections import defaultdict
from rustworkx import PyDAG, topological_sort, number_weakly_connected_components

from qiskit.pulse.compiler.basepasses import TransformationPass
from qiskit.pulse.ir import SequenceIR
from qiskit.pulse.transforms import AlignmentKind, AlignLeft, AlignRight, AlignSequential
from qiskit.pulse.exceptions import PulseCompilerError


class SetSchedule(TransformationPass):
    """Concretely schedule ``SequenceIR`` object.

    The pass traverses the ``SequenceIR``, and recursively sets initial time for every
    node in the sequence (and sub-sequences). The scheduling is done according to the
    alignment strategy, and requires that the ``sequence`` property is already sequenced,
    typically with the pass :class:`~qiskit.pulse.compiler.passes.SetSequence`."""

    def __init__(self):
        """Create new SetSchedule pass"""
        super().__init__(target=None)

    def run(
        self,
        passmanager_ir: SequenceIR,
    ) -> SequenceIR:

        self._schedule_recursion(passmanager_ir)
        return passmanager_ir

    def _schedule_recursion(self, prog: SequenceIR) -> None:
        """Recursively schedule the IR.

        Nested IR objects must be scheduled first, so we traverse the IR,
        and recursively schedule the IR objects.
        After all nested IR objects are scheduled, we apply the scheduling strategy to the
        current object.

        Arguments:
            prog: The IR object to be scheduled.
        """
        for elem in prog.elements():
            if isinstance(elem, SequenceIR):
                self._schedule_recursion(elem)

        self._schedule_single_program(prog.alignment, prog.sequence, prog.time_table)

    @singledispatchmethod
    def _schedule_single_program(
        self, alignment: AlignmentKind, sequence: PyDAG, time_table: defaultdict
    ) -> None:
        """Concretely schedule the IR object.

        The ``time_table`` argument is mutated to include the initial time of each element of
        ``sequence``, according to the structure of ``sequence`` and the alignment.
        The function assumes that nested IR objects are already scheduled.

        ``sequence`` is assumed to have the following structure - node 0 marks the beginning of the
        sequence, while node 1 marks the end of it. All branches of the graph originate from node 0
        and end at node 1.

        Arguments:
            alignment: The alignment of the IR object.
            sequence: The sequence of the IR object.
            time_table: The time_table of the IR object.
        """
        raise NotImplementedError

    # pylint: disable=unused-argument
    @_schedule_single_program.register(AlignLeft)
    @_schedule_single_program.register(AlignSequential)
    def _schedule_asap(
        self, alignment: AlignmentKind, sequence: PyDAG, time_table: defaultdict
    ) -> None:
        """Concretely schedule the IR object, aligning to the left.

        The ``time_table`` argument is mutated to include the initial time of each element of
        ``sequence``, according to the structure of ``sequence`` and aligning to the left.
        The function assumes that nested IR objects are already scheduled.

        ``sequence`` is assumed to have the following structure - node 0 marks the beginning of the
        sequence, while node 1 marks the end of it. All branches of the graph originate from node 0
        and end at node 1.

        Arguments:
            alignment: The alignment of the IR object.
            sequence: The sequence of the IR object.
            time_table: The time_table of the IR object.

        Raises:
            PulseCompilerError: If the sequence is not sequenced as expected.
        """
        nodes = topological_sort(sequence)
        if number_weakly_connected_components(sequence) != 1 or nodes[0] != 0 or nodes[-1] != 1:
            raise PulseCompilerError(
                "The pulse program is not sequenced as expected. "
                "Insert SetSequence pass in advance of SchedulePass."
            )

        for node_index in nodes:
            if node_index in (0, 1):
                # in,out nodes
                continue
            preds = sequence.predecessor_indices(node_index)
            if preds == [0]:
                time_table[node_index] = 0
            else:
                time_table[node_index] = max(
                    time_table[pred] + sequence.get_node_data(pred).duration for pred in preds
                )

    # pylint: disable=unused-argument
    @_schedule_single_program.register(AlignRight)
    def _schedule_alap(
        self, alignment: AlignmentKind, sequence: PyDAG, time_table: defaultdict
    ) -> None:
        """Concretely schedule the IR object, aligning to the right.

        The ``time_table`` argument is mutated to include the initial time of each element of
        ``sequence``, according to the structure of ``sequence`` and aligning to the right.
        The function assumes that nested IR objects are already scheduled.

        ``sequence`` is assumed to have the following structure - node 0 marks the beginning of the
        sequence, while node 1 marks the end of it. All branches of the graph originate from node 0
        and end at node 1.

        Arguments:
            alignment: The alignment of the IR object.
            sequence: The sequence of the IR object.
            time_table: The time_table of the IR object.

        Raises:
            PulseCompilerError: If the sequence is not sequenced as expected.
        """
        # We reverse the sequence, schedule to the left, then reverse the timings.
        reversed_sequence = sequence.copy()
        reversed_sequence.reverse()

        nodes = topological_sort(reversed_sequence)

        if number_weakly_connected_components(sequence) != 1 or nodes[0] != 1 or nodes[-1] != 0:
            raise PulseCompilerError(
                "The pulse program is not sequenced as expected. "
                "Insert SetSequence pass in advance of SchedulePass."
            )

        for node_index in nodes:
            if node_index in (0, 1):
                # in,out nodes
                continue
            preds = reversed_sequence.predecessor_indices(node_index)
            if preds == [1]:
                time_table[node_index] = 0
            else:
                time_table[node_index] = max(
                    time_table[pred] + sequence.get_node_data(pred).duration for pred in preds
                )

        total_duration = max(
            time_table[i] + sequence.get_node_data(i).duration
            for i in reversed_sequence.predecessor_indices(0)
        )

        for node in sequence.node_indices():
            if node not in (0, 1):
                time_table[node] = (
                    total_duration - time_table[node] - sequence.get_node_data(node).duration
                )

    def __hash__(self):
        return hash((self.__class__.__name__,))

    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__
