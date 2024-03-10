# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""A collection of passes to reallocate the timeslots of instructions according to context."""
from __future__ import annotations
from collections import defaultdict
import abc
from typing import Callable, Tuple
from rustworkx import PyDAG

import numpy as np

from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ScheduleComponent
from qiskit.pulse.utils import instruction_duration_validation
from qiskit.pulse.model import MixedFrame


class AlignmentKind(abc.ABC):
    """An abstract class for schedule alignment."""

    def __init__(self, context_params: Tuple[ParameterValueType, ...]):
        """Create new context."""
        self._context_params = tuple(context_params)

    @abc.abstractmethod
    def align(self, schedule: Schedule) -> Schedule:
        """Reallocate instructions according to the policy.

        Only top-level sub-schedules are aligned. If sub-schedules are nested,
        nested schedules are not recursively aligned.

        Args:
            schedule: Schedule to align.

        Returns:
            Schedule with reallocated instructions.
        """
        pass

    @abc.abstractmethod
    def set_sequence(self, sequence: PyDAG, **kwargs) -> None:
        """Set the sequence of the IR program according to the policy.

        The ``sequence`` is mutated to include all the edges
        connecting the elements in the sequence.

        Only top-level elements are sequences. If sub-sequences are nested,
        nested sequences are not recursively set.

        Args:
            sequence: The graph object to be sequenced.
            kwargs: Keyword arguments needed for some subclasses.
        """
        pass

    @abc.abstractmethod
    def schedule(self, sequence: PyDAG, time_table: defaultdict) -> None:
        """Concretely schedule a ``SequenceIR`` program.

        Only top-level elements are scheduled, and sub-sequences are assumed to have been
        scheduled beforehand (because scheduling of parent sequence depends on the duration
         - and thus scheduling - of child sequences).

        The ``time_table`` dictionary will be mutated to include  the initial time of each node
        in the sequence, keyed on the node index.

        Args:
            sequence: The graph object to be scheduled.
            time_table: A default dictionary (with default value ``None``) to be mutated.
        """
        pass

    @property
    @abc.abstractmethod
    def is_sequential(self) -> bool:
        """Return ``True`` if this is sequential alignment context.

        This information is used to evaluate DAG equivalency of two :class:`.ScheduleBlock`s.
        When the context has two pulses in different channels,
        a sequential context subtype intends to return following scheduling outcome.

        .. parsed-literal::

                ┌────────┐
            D0: ┤ pulse1 ├────────────
                └────────┘  ┌────────┐
            D1: ────────────┤ pulse2 ├
                            └────────┘

        On the other hand, parallel context with ``is_sequential=False`` returns

        .. parsed-literal::

                ┌────────┐
            D0: ┤ pulse1 ├
                ├────────┤
            D1: ┤ pulse2 ├
                └────────┘

        All subclasses must implement this method according to scheduling strategy.
        """
        pass

    def __eq__(self, other: object) -> bool:
        """Check equality of two transforms."""
        if type(self) is not type(other):
            return False
        if self._context_params != other._context_params:
            return False
        return True

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(self._context_params)})"


class SequentialAlignment(AlignmentKind, abc.ABC):
    """Abstract base class for ``AlignmentKind`` which aligns instructions sequentially"""

    def set_sequence(self, sequence: PyDAG, **kwargs) -> None:
        """Sets the sequence sequentially.

        The ``sequence`` property of ``ir_program`` is mutated to include all the edges
        connecting the elements of the sequence sequentially according to their order.

        Only top-level elements are sequences. If sub-sequences are nested,
        nested sequences are not recursively set.

        Args:
            sequence: The graph object to be sequenced.
            kwargs: Included only to match the signature of the function with other subclasses.
        """
        nodes = sequence.node_indices()
        prev = 0
        # The first two nodes are the in\out nodes.
        for ind in nodes[2:]:
            sequence.add_edge(prev, ind, None)
            prev = ind
        sequence.add_edge(prev, 1, None)

    @property
    def is_sequential(self) -> bool:
        return True


class ParallelAlignment(AlignmentKind, abc.ABC):
    """Abstract base class for ``AlignmentKind`` which aligns instructions in parallel"""

    def set_sequence(self, sequence: PyDAG, **kwargs) -> None:
        """Sets the sequence in parallel.

        The ``sequence`` property of ``ir_program`` is mutated to include all the edges
        connecting the elements of the sequence in parallel.

        The function requires a ``mixed_frame_mapping`` dictionary - mapping all ``PulseTarget``
        and ``Frame`` to the associated ``MixedFrame`` - to be passed to ``kwargs``. As part
        of a pass manager work flow the dictionary is obtained via the pass
        :class:`~qiskit.pulse.compiler.MapMixedFrames`.

        Only top-level elements are sequenced. If sub-sequences are nested,
        nested sequences are not recursively set.

        Args:
            sequence: The graph object to be sequenced.
            kwargs: Expecting a keyword argument ``mixed_frame_mapping``.

        Raises:
            PulseError: if ``kwargs`` does not include a "mixed_frames_mapping" key.
        """
        if "mixed_frames_mapping" not in kwargs.keys() or kwargs["mixed_frames_mapping"] is None:
            raise PulseError(
                "Expected a keyword argument mixed_frames_mapping with a"
                " mapping of PulseTarget and Frame to MixedFrame"
            )
        mixed_frame_mapping = kwargs["mixed_frames_mapping"]

        idle_after = {}
        for ind in sequence.node_indices():
            if ind in (0, 1):
                # In, Out node
                continue
            node = sequence.get_node_data(ind)
            node_mixed_frames = set()

            # if isinstance(node, SequenceIR):
            #     inst_targets = node.inst_targets
            # else:
            #     inst_targets = [node.inst_target]
            if hasattr(node, "inst_targets"):
                # SequenceIR object
                inst_targets = node.inst_targets
            else:
                # Instruction object
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

    @property
    def is_sequential(self) -> bool:
        return False


class AlignLeft(ParallelAlignment):
    """Align instructions in as-soon-as-possible manner.

    Instructions are placed at earliest available timeslots.
    """

    def __init__(self):
        """Create new left-justified context."""
        super().__init__(context_params=())

    def align(self, schedule: Schedule) -> Schedule:
        """Reallocate instructions according to the policy.

        Only top-level sub-schedules are aligned. If sub-schedules are nested,
        nested schedules are not recursively aligned.

        Args:
            schedule: Schedule to align.

        Returns:
            Schedule with reallocated instructions.
        """
        aligned = Schedule.initialize_from(schedule)
        for _, child in schedule.children:
            self._push_left_append(aligned, child)

        return aligned

    @staticmethod
    def _push_left_append(this: Schedule, other: ScheduleComponent) -> Schedule:
        """Return ``this`` with ``other`` inserted at the maximum time over
        all channels shared between ```this`` and ``other``.

        Args:
            this: Input schedule to which ``other`` will be inserted.
            other: Other schedule to insert.

        Returns:
            Push left appended schedule.
        """
        this_channels = set(this.channels)
        other_channels = set(other.channels)
        shared_channels = list(this_channels & other_channels)
        ch_slacks = [
            this.stop_time - this.ch_stop_time(channel) + other.ch_start_time(channel)
            for channel in shared_channels
        ]

        if ch_slacks:
            slack_chan = shared_channels[np.argmin(ch_slacks)]
            shared_insert_time = this.ch_stop_time(slack_chan) - other.ch_start_time(slack_chan)
        else:
            shared_insert_time = 0

        # Handle case where channels not common to both might actually start
        # after ``this`` has finished.
        other_only_insert_time = other.ch_start_time(*(other_channels - this_channels))
        # Choose whichever is greatest.
        insert_time = max(shared_insert_time, other_only_insert_time)

        return this.insert(insert_time, other, inplace=True)

    def schedule(self, sequence: PyDAG, time_table: defaultdict) -> None:
        """Concretely schedule a ``SequenceIR`` program with left alignment.

        Only top-level elements are scheduled, and sub-sequences are assumed to have been
        scheduled beforehand (because scheduling of parent sequence depends on the duration
         - and thus scheduling - of child sequences).

        The ``time_table`` dictionary will be mutated to include  the initial time of each node
        in the sequence, keyed on the node index.

        Args:
            sequence: The graph object to be scheduled.
            time_table: A default dictionary (with default value ``None``) to be mutated.
        """
        first_nodes = sequence.successor_indices(0)
        active_nodes = []
        # Node 0 has no duration so is treated separately
        for node_index in first_nodes:
            time_table[node_index] = 0
            active_nodes.extend(sequence.successor_indices(node_index))

        count = 0
        max_count = sequence.num_nodes() * 10

        while active_nodes and count < max_count:
            count += 1
            node_index = active_nodes.pop(0)
            if node_index == 1 or time_table[node_index] is not None:
                # reached end or already scheduled
                continue
            preds = sequence.predecessor_indices(node_index)
            if all(time_table[pred] is not None for pred in preds):
                # All predecessors are scheduled, we can schedule this node.
                t0 = max(time_table[pred] + sequence.get_node_data(pred).duration for pred in preds)
                time_table[node_index] = t0
                active_nodes.extend(sequence.successor_indices(node_index))
            else:
                # Predecessors not scheduled, we move this node to the back of the line.
                active_nodes.append(node_index)

        if len(active_nodes) != 0:
            raise PulseError("Could not schedule the sequence. Infinite loop created.")


class AlignRight(ParallelAlignment):
    """Align instructions in as-late-as-possible manner.

    Instructions are placed at latest available timeslots.
    """

    def __init__(self):
        """Create new right-justified context."""
        super().__init__(context_params=())

    def align(self, schedule: Schedule) -> Schedule:
        """Reallocate instructions according to the policy.

        Only top-level sub-schedules are aligned. If sub-schedules are nested,
        nested schedules are not recursively aligned.

        Args:
            schedule: Schedule to align.

        Returns:
            Schedule with reallocated instructions.
        """
        aligned = Schedule.initialize_from(schedule)
        for _, child in reversed(schedule.children):
            aligned = self._push_right_prepend(aligned, child)

        return aligned

    @staticmethod
    def _push_right_prepend(this: Schedule, other: ScheduleComponent) -> Schedule:
        """Return ``this`` with ``other`` inserted at the latest possible time
        such that ``other`` ends before it overlaps with any of ``this``.

        If required ``this`` is shifted  to start late enough so that there is room
        to insert ``other``.

        Args:
           this: Input schedule to which ``other`` will be inserted.
           other: Other schedule to insert.

        Returns:
           Push right prepended schedule.
        """
        this_channels = set(this.channels)
        other_channels = set(other.channels)
        shared_channels = list(this_channels & other_channels)
        ch_slacks = [
            this.ch_start_time(channel) - other.ch_stop_time(channel) for channel in shared_channels
        ]

        if ch_slacks:
            insert_time = min(ch_slacks) + other.start_time
        else:
            insert_time = this.stop_time - other.stop_time + other.start_time

        if insert_time < 0:
            this.shift(-insert_time, inplace=True)
            this.insert(0, other, inplace=True)
        else:
            this.insert(insert_time, other, inplace=True)

        return this

    def schedule(self, sequence: PyDAG, time_table: defaultdict) -> None:
        """Concretely schedule a ``SequenceIR`` program with right alignment.

        Only top-level elements are scheduled, and sub-sequences are assumed to have been
        scheduled beforehand (because scheduling of parent sequence depends on the duration
         - and thus scheduling - of child sequences).

        The ``time_table`` dictionary will be mutated to include the initial time of each node
        in the sequence, keyed on the node index.

        Args:
            sequence: The graph object to be scheduled.
            time_table: A default dictionary (with default value ``None``) to be mutated.
        """
        # We reverse the sequence, align left, and then flip the timings.

        reversed_sequence = sequence.copy()
        # Reverse all edges
        reversed_sequence.reverse()
        # Now swap 0 and 1 nodes
        new_start_node_successors = reversed_sequence.successor_indices(1)
        new_end_node_predecessors = reversed_sequence.predecessor_indices(0)
        reversed_sequence.remove_node(0)
        reversed_sequence.remove_node(1)

        # Note that these are not the same node objects typically in SequenceIR.
        reversed_sequence.add_node(None)
        reversed_sequence.add_node(None)

        reversed_sequence.add_edges_from_no_data([(0, x) for x in new_start_node_successors])
        reversed_sequence.add_edges_from_no_data([(x, 1) for x in new_end_node_predecessors])

        # Schedule with left alignment
        AlignLeft().schedule(reversed_sequence, time_table)

        # Then reverse the timings
        total_duration = max(
            time_table[i] + reversed_sequence.get_node_data(i).duration
            for i in reversed_sequence.predecessor_indices(1)
        )
        for node in sequence.node_indices():
            if node not in (0, 1):
                time_table[node] = (
                    total_duration
                    - time_table[node]
                    - reversed_sequence.get_node_data(node).duration
                )


class AlignSequential(SequentialAlignment):
    """Align instructions sequentially.

    Instructions played on different channels are also arranged in a sequence.
    No buffer time is inserted in between instructions.
    """

    def __init__(self):
        """Create new sequential context."""
        super().__init__(context_params=())

    def align(self, schedule: Schedule) -> Schedule:
        """Reallocate instructions according to the policy.

        Only top-level sub-schedules are aligned. If sub-schedules are nested,
        nested schedules are not recursively aligned.

        Args:
            schedule: Schedule to align.

        Returns:
            Schedule with reallocated instructions.
        """
        aligned = Schedule.initialize_from(schedule)
        for _, child in schedule.children:
            aligned.insert(aligned.duration, child, inplace=True)

        return aligned

    def schedule(self, sequence: PyDAG, time_table: defaultdict) -> None:
        """Concretely schedule a ``SequenceIR`` program with sequential alignment.

        Only top-level elements are scheduled, and sub-sequences are assumed to have been
        scheduled beforehand (because scheduling of parent sequence depends on the duration
         - and thus scheduling - of child sequences).

        The ``time_table`` dictionary will be mutated to include the initial time of each node
        in the sequence, keyed on the node index.

        Args:
            sequence: The graph object to be scheduled.
            time_table: A default dictionary (with default value ``None``) to be mutated.
        """
        # In terms of scheduling, sequential alignment is the same as left alignment.
        # (The two differ in sequencing)
        AlignLeft().schedule(sequence, time_table)


class AlignEquispaced(SequentialAlignment):
    """Align instructions with equispaced interval within a specified duration.

    Instructions played on different channels are also arranged in a sequence.
    This alignment is convenient to create dynamical decoupling sequences such as PDD.
    """

    def __init__(self, duration: int | ParameterExpression):
        """Create new equispaced context.

        Args:
            duration: Duration of this context. This should be larger than the schedule duration.
                If the specified duration is shorter than the schedule duration,
                no alignment is performed and the input schedule is just returned.
                This duration can be parametrized.
        """
        super().__init__(context_params=(duration,))

    @property
    def duration(self):
        """Return context duration."""
        return self._context_params[0]

    def align(self, schedule: Schedule) -> Schedule:
        """Reallocate instructions according to the policy.

        Only top-level sub-schedules are aligned. If sub-schedules are nested,
        nested schedules are not recursively aligned.

        Args:
            schedule: Schedule to align.

        Returns:
            Schedule with reallocated instructions.
        """
        instruction_duration_validation(self.duration)

        total_duration = sum(child.duration for _, child in schedule.children)
        if self.duration < total_duration:
            return schedule

        total_delay = self.duration - total_duration

        if len(schedule.children) > 1:
            # Calculate the interval in between sub-schedules.
            # If the duration cannot be divided by the number of sub-schedules,
            # the modulo is appended and prepended to the input schedule.
            interval, mod = np.divmod(total_delay, len(schedule.children) - 1)
        else:
            interval = 0
            mod = total_delay

        # Calculate pre schedule delay
        delay, mod = np.divmod(mod, 2)

        aligned = Schedule.initialize_from(schedule)
        # Insert sub-schedules with interval
        _t0 = int(aligned.stop_time + delay + mod)
        for _, child in schedule.children:
            aligned.insert(_t0, child, inplace=True)
            _t0 = int(aligned.stop_time + interval)

        return aligned

    def schedule(self, sequence: PyDAG, time_table: defaultdict) -> None:
        """Concretely schedule a ``SequenceIR`` program with equisapced alignment.

        Only top-level elements are scheduled, and sub-sequences are assumed to have been
        scheduled beforehand (because scheduling of parent sequence depends on the duration
         - and thus scheduling - of child sequences).

        The ``time_table`` dictionary will be mutated to include  the initial time of each node
        in the sequence, keyed on the node index.

        Args:
            sequence: The graph object to be scheduled.
            time_table: A default dictionary (with default value ``None``) to be mutated.
        """
        raise NotImplementedError
        # TODO : Implement this.


class AlignFunc(SequentialAlignment):
    """Allocate instructions at position specified by callback function.

    The position is specified for each instruction of index ``j`` as a
    fractional coordinate in [0, 1] within the specified duration.

    Instructions played on different channels are also arranged in a sequence.
    This alignment is convenient to create dynamical decoupling sequences such as UDD.

    For example, UDD sequence with 10 pulses can be specified with following function.

    .. code-block:: python

        def udd10_pos(j):
        return np.sin(np.pi*j/(2*10 + 2))**2

    .. note::

        This context cannot be QPY serialized because of the callable. If you use this context,
        your program cannot be saved in QPY format.

    """

    def __init__(self, duration: int | ParameterExpression, func: Callable):
        """Create new equispaced context.

        Args:
            duration: Duration of this context. This should be larger than the schedule duration.
                If the specified duration is shorter than the schedule duration,
                no alignment is performed and the input schedule is just returned.
                This duration can be parametrized.
            func: A function that takes an index of sub-schedule and returns the
                fractional coordinate of of that sub-schedule. The returned value should be
                defined within [0, 1]. The pulse index starts from 1.
        """
        super().__init__(context_params=(duration, func))

    @property
    def duration(self):
        """Return context duration."""
        return self._context_params[0]

    @property
    def func(self):
        """Return context alignment function."""
        return self._context_params[1]

    def align(self, schedule: Schedule) -> Schedule:
        """Reallocate instructions according to the policy.

        Only top-level sub-schedules are aligned. If sub-schedules are nested,
        nested schedules are not recursively aligned.

        Args:
            schedule: Schedule to align.

        Returns:
            Schedule with reallocated instructions.
        """
        instruction_duration_validation(self.duration)

        if self.duration < schedule.duration:
            return schedule

        aligned = Schedule.initialize_from(schedule)
        for ind, (_, child) in enumerate(schedule.children):
            _t_center = self.duration * self.func(ind + 1)
            _t0 = int(_t_center - 0.5 * child.duration)
            if _t0 < 0 or _t0 > self.duration:
                raise PulseError(
                    "Invalid schedule position t=%d is specified at index=%d" % (_t0, ind)
                )
            aligned.insert(_t0, child, inplace=True)

        return aligned

    def schedule(self, sequence: PyDAG, time_table: defaultdict) -> None:
        """Concretely schedule a ``SequenceIR`` program with functional alignment.

        Only top-level elements are scheduled, and sub-sequences are assumed to have been
        scheduled beforehand (because scheduling of parent sequence depends on the duration
         - and thus scheduling - of child sequences).

        The ``time_table`` dictionary will be mutated to include  the initial time of each node
        in the sequence, keyed on the node index.

        Args:
            sequence: The graph object to be scheduled.
            time_table: A default dictionary (with default value ``None``) to be mutated.
        """
        raise NotImplementedError
        # TODO : Implement this.
