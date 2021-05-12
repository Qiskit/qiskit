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

import abc
import copy
from typing import Callable, Dict, Any, Union, Iterator

import numpy as np

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ScheduleBlock, ScheduleNode
from qiskit.pulse.instructions import RelativeBarrier
from qiskit.pulse.utils import instruction_duration_validation, deprecated_functionality


class AlignmentKind(abc.ABC):
    """An abstract class for schedule alignment."""

    is_sequential = None

    def __init__(self):
        """Create new context."""
        self._context_params = tuple()

    @abc.abstractmethod
    def align(
        self, schedule: Union[Schedule, ScheduleBlock], inplace: bool = False
    ) -> Union[Schedule, ScheduleBlock]:
        """Reallocate instructions according to the policy.

        Only top-level sub-schedules are aligned. If sub-schedules are nested,
        nested schedules are not recursively aligned.

        Args:
            schedule: Schedule to align.

        Returns:
            Schedule with reallocated instructions.
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Returns dictionary to represent this alignment."""
        return {"alignment": self.__class__.__name__}

    def __eq__(self, other):
        """Check equality of two transforms."""
        return isinstance(other, type(self)) and self.to_dict() == other.to_dict()

    def __repr__(self):
        name = self.__class__.__name__
        opts = self.to_dict()
        opts.pop("alignment")
        opts_str = ", ".join(f"{key}={val}" for key, val in opts.items())
        return f"{name}({opts_str})"


class Frozen(AlignmentKind):
    """Keep node time."""

    is_sequential = False

    def align(
        self, schedule: Union[Schedule, ScheduleBlock], inplace: bool = False
    ) -> Union[Schedule, ScheduleBlock]:
        """Reallocate instructions according to the policy.

        Only top-level sub-schedules are aligned. If sub-schedules are nested,
        nested schedules are not recursively aligned.

        Args:
            schedule: Schedule to align.
            inplace: Set ``True`` to override input schedule.

        Returns:
            Schedule with reallocated instructions.
        """
        if inplace:
            return schedule
        else:
            return copy.deepcopy(schedule)


class AlignLeft(AlignmentKind):
    """Align instructions in as-soon-as-possible manner.

    Instructions are placed at earliest available timeslots.
    """

    is_sequential = False

    def align(
        self, schedule: Union[Schedule, ScheduleBlock], inplace: bool = False
    ) -> Union[Schedule, ScheduleBlock]:
        """Reallocate instructions according to the policy.

        Only top-level sub-schedules are aligned. If sub-schedules are nested,
        nested schedules are not recursively aligned.

        Args:
            schedule: Schedule to align.
            inplace: Set ``True`` to override input schedule.

        Returns:
            Schedule with reallocated instructions.
        """
        if not inplace:
            aligned = copy.deepcopy(schedule)
        else:
            aligned = schedule

        # this should not have logic accessing to timeslot, i.e. block has no timeslot.
        ch_stop_time = {chan: 0 for chan in aligned.channels}
        for node in _time_ordered_node(aligned):
            op_data = node.data
            t_start = max([ch_stop_time[chan] for chan in op_data.channels])
            node.time = t_start
            t_stop = t_start + op_data.duration
            for chan in op_data.channels:
                ch_stop_time[chan] = t_stop

        if isinstance(schedule, Schedule):
            aligned._renew_timeslots()

        return aligned


class AlignRight(AlignmentKind):
    """Align instructions in as-late-as-possible manner.

    Instructions are placed at latest available timeslots.
    """

    is_sequential = False

    def align(
        self, schedule: Union[Schedule, ScheduleBlock], inplace: bool = False
    ) -> Union[Schedule, ScheduleBlock]:
        """Reallocate instructions according to the policy.

        Only top-level sub-schedules are aligned. If sub-schedules are nested,
        nested schedules are not recursively aligned.

        Args:
            schedule: Schedule to align.
            inplace: Set ``True`` to override input schedule.

        Returns:
            Schedule with reallocated instructions.
        """
        if not inplace:
            aligned = copy.deepcopy(schedule)
        else:
            aligned = schedule

        # this should not have logic accessing to timeslot, i.e. block has no timeslot.
        ch_stop_time = {chan: 0 for chan in aligned.channels}
        for node in _time_ordered_node(aligned, reversed=True):
            op_data = node.data
            t_stop = min([ch_stop_time[chan] for chan in op_data.channels])
            t_start = t_stop - op_data.duration
            node.time = t_start
            for chan in op_data.channels:
                ch_stop_time[chan] = t_start

        # shift to positive time
        duration = min(ch_stop_time.values())
        for node in aligned.nodes():
            node.time -= duration

        if isinstance(schedule, Schedule):
            aligned._renew_timeslots()
        else:
            aligned._duration = int(np.abs(duration))

        return aligned


class AlignSequential(AlignmentKind):
    """Align instructions sequentially.

    Instructions played on different channels are also arranged in a sequence.
    No buffer time is inserted in between instructions.
    """

    is_sequential = True

    def align(
        self, schedule: Union[Schedule, ScheduleBlock], inplace: bool = False
    ) -> Union[Schedule, ScheduleBlock]:
        """Reallocate instructions according to the policy.

        Only top-level sub-schedules are aligned. If sub-schedules are nested,
        nested schedules are not recursively aligned.

        Args:
            schedule: Schedule to align.
            inplace: Set ``True`` to override input schedule.

        Returns:
            Schedule with reallocated instructions.
        """
        if not inplace:
            aligned = copy.deepcopy(schedule)
        else:
            aligned = schedule

        # this should not have logic accessing to timeslot, i.e. block has no timeslot.
        last_t0 = 0
        for node in _time_ordered_node(aligned):
            node.time = last_t0
            last_t0 += node.data.duration

        if isinstance(schedule, Schedule):
            aligned._renew_timeslots()

        return aligned


class AlignEquispaced(AlignmentKind):
    """Align instructions with equispaced interval within a specified duration.

    Instructions played on different channels are also arranged in a sequence.
    This alignment is convenient to create dynamical decoupling sequences such as PDD.
    """

    is_sequential = True

    def __init__(self, duration: Union[int, ParameterExpression]):
        """Create new equispaced context.

        Args:
            duration: Duration of this context. This should be larger than the schedule duration.
                This duration can be parametrized.
        """
        super().__init__()

        self._context_params = (duration,)

    @property
    def duration(self):
        """Return context duration."""
        return self._context_params[0]

    def align(
        self, schedule: Union[Schedule, ScheduleBlock], inplace: bool = False
    ) -> Union[Schedule, ScheduleBlock]:
        """Reallocate instructions according to the policy.

        Only top-level sub-schedules are aligned. If sub-schedules are nested,
        nested schedules are not recursively aligned.

        Args:
            schedule: Schedule to align.
            inplace: Set ``True`` to override input schedule.

        Returns:
            Schedule with reallocated instructions.

        Raises:
            PulseError: When context duration is shorter than total schedule duration.
        """
        instruction_duration_validation(self.duration)

        if not inplace:
            aligned = copy.deepcopy(schedule)
        else:
            aligned = schedule

        total_duration = sum([node.data.duration for node in schedule.nodes()])
        if self.duration < total_duration:
            raise PulseError(
                "Context duration is shorter than minimum schedule duration. "
                f"{self.duration} < {total_duration}. This operation cannot be performed."
            )
        total_delay = self.duration - total_duration

        num_nodes = len(schedule)
        if num_nodes > 1:
            # Calculate the interval in between sub-schedules.
            # If the duration cannot be divided by the number of sub-schedules,
            # the modulo is appended and prepended to the input schedule.
            interval, mod = np.divmod(total_delay, num_nodes - 1)
        else:
            interval = 0
            mod = total_delay

        # Calculate pre schedule delay
        pre_delay, mod = np.divmod(mod, 2)

        # this should not have logic accessing to timeslot, i.e. block has no timeslot.
        last_t0 = pre_delay
        for node in _time_ordered_node(aligned):
            node.time = last_t0
            last_t0 += node.data.duration + interval

        # to prevent interruption of following instruction.
        aligned.add_node(op_data=RelativeBarrier(*aligned.channels), time=self.duration)

        if isinstance(schedule, Schedule):
            aligned._renew_timeslots()

        return aligned

    def to_dict(self) -> Dict[str, Any]:
        """Returns dictionary to represent this alignment."""
        return {"alignment": self.__class__.__name__, "duration": self.duration}


class AlignFunc(AlignmentKind):
    """Allocate instructions at position specified by callback function.

    The position is specified for each instruction of index ``j`` as a
    fractional coordinate in [0, 1] within the specified duration.

    Instructions played on different channels are also arranged in a sequence.
    This alignment is convenient to create dynamical decoupling sequences such as UDD.

    For example, UDD sequence with 10 pulses can be specified with following function.

    .. code-block:: python

        def udd10_pos(j):
        return np.sin(np.pi*j/(2*10 + 2))**2
    """

    is_sequential = True

    def __init__(self, duration: Union[int, ParameterExpression], func: Callable):
        """Create new equispaced context.

        Args:
            duration: Duration of this context. This should be larger than the schedule duration.
                This duration can be parametrized.
            func: A function that takes an index of sub-schedule and returns the
                fractional coordinate of of that sub-schedule. The returned value should be
                defined within [0, 1]. The pulse index starts from 1.
        """
        super().__init__()

        self._context_params = (duration,)
        self._func = func

    @property
    def duration(self):
        """Return context duration."""
        return self._context_params[0]

    def align(
        self, schedule: Union[Schedule, ScheduleBlock], inplace: bool = False
    ) -> Union[Schedule, ScheduleBlock]:
        """Reallocate instructions according to the policy.

        Only top-level sub-schedules are aligned. If sub-schedules are nested,
        nested schedules are not recursively aligned.

        Args:
            schedule: Schedule to align.
            inplace: Set ``True`` to override input schedule.

        Returns:
            Schedule with reallocated instructions.

        Raises:
            PulseError:
                - When context duration is shorter than total schedule duration.
                - When i-th location p_i = f(i) is too small or too large.
        """
        instruction_duration_validation(self.duration)

        if not inplace:
            aligned = copy.deepcopy(schedule)
        else:
            aligned = schedule

        total_duration = sum([node.data.duration for node in schedule.nodes()])
        if self.duration < total_duration:
            raise PulseError(
                "Context duration is shorter than minimum schedule duration. "
                f"{self.duration} < {total_duration}. This operation cannot be performed."
            )

        # this should not have logic accessing to timeslot, i.e. block has no timeslot.
        for i, node in enumerate(_time_ordered_node(aligned)):
            location = self.duration * self._func(i + 1)
            t0 = int(location - 0.5 * node.data.duration)
            if t0 < 0 or t0 > self.duration:
                PulseError("Invalid t0=%d is specified at index=%d" % (t0, i))
            node.time = t0

        # to prevent interruption of following instruction.
        aligned.add_node(op_data=RelativeBarrier(*aligned.channels), time=self.duration)

        if isinstance(schedule, Schedule):
            aligned._renew_timeslots()

        return aligned

    def to_dict(self) -> Dict[str, Any]:
        """Returns dictionary to represent this alignment.

        .. note:: ``func`` is not presented in this dictionary. Just name.
        """
        return {
            "alignment": self.__class__.__name__,
            "duration": self._context_params[0],
            "func": self._func.__name__,
        }


def _time_ordered_node(
    schedule: Union[Schedule, ScheduleBlock], reversed: bool = False
) -> Iterator[ScheduleNode]:
    """A helper function to return a generator of time-ordered program node.

    .. notes::
        This function returns mutable nodes.

    Args:
        schedule: Target schedule to return nodes.
        reversed: Set ``True`` to return node in reversed order.
    """
    try:
        nodes = sorted(schedule.nodes(), key=lambda node: node.time)
    except TypeError:
        # sort time assigned node while keeping the position of parametrized duration node
        scheduled_nodes = []
        flexible_nodes = []

        prev_node = None
        for node in schedule.nodes():
            if node.time is None:
                flexible_nodes.append((prev_node, node))
            else:
                scheduled_nodes.append(node)
            prev_node = node.location

        scheduled_nodes = sorted(scheduled_nodes, key=lambda node: node.time)

        # insert flexible node into right position
        # we can track node position thanks to unique location attribute.
        for location, node in flexible_nodes:
            if location is None:
                scheduled_nodes.insert(0, node)
            else:
                list_pos = scheduled_nodes.index(location) + 1
                scheduled_nodes.insert(list_pos, node)
        nodes = scheduled_nodes

    if reversed:
        yield from nodes[::-1]
    else:
        yield from nodes


@deprecated_functionality
def align_left(schedule: Schedule) -> Schedule:
    """Align a list of pulse instructions on the left.

    Args:
        schedule: Input schedule of which top-level sub-schedules will be rescheduled.

    Returns:
        New schedule with input `schedule`` child schedules and instructions
        left aligned.
    """
    context = AlignLeft()
    return context.align(schedule)


@deprecated_functionality
def align_right(schedule: Schedule) -> Schedule:
    """Align a list of pulse instructions on the right.

    Args:
        schedule: Input schedule of which top-level sub-schedules will be rescheduled.

    Returns:
        New schedule with input `schedule`` child schedules and instructions
        right aligned.
    """
    context = AlignRight()
    return context.align(schedule)


@deprecated_functionality
def align_sequential(schedule: Schedule) -> Schedule:
    """Schedule all top-level nodes in parallel.

    Args:
        schedule: Input schedule of which top-level sub-schedules will be rescheduled.

    Returns:
        New schedule with input `schedule`` child schedules and instructions
        applied sequentially across channels
    """
    context = AlignSequential()
    return context.align(schedule)


@deprecated_functionality
def align_equispaced(schedule: Schedule, duration: int) -> Schedule:
    """Schedule a list of pulse instructions with equivalent interval.

    Args:
        schedule: Input schedule of which top-level sub-schedules will be rescheduled.
        duration: Duration of context. This should be larger than the schedule duration.

    Returns:
        New schedule with input `schedule`` child schedules and instructions
        aligned with equivalent interval.

    Notes:
        This context is convenient for writing PDD or Hahn echo sequence for example.
    """
    context = AlignEquispaced(duration=duration)
    return context.align(schedule)


@deprecated_functionality
def align_func(schedule: Schedule, duration: int, func: Callable[[int], float]) -> Schedule:
    """Schedule a list of pulse instructions with schedule position defined by the
    numerical expression.

    Args:
        schedule: Input schedule of which top-level sub-schedules will be rescheduled.
        duration: Duration of context. This should be larger than the schedule duration.
        func: A function that takes an index of sub-schedule and returns the
            fractional coordinate of of that sub-schedule.
            The returned value should be defined within [0, 1].
            The pulse index starts from 1.

    Returns:
        New schedule with input `schedule`` child schedules and instructions
        aligned with equivalent interval.

    Notes:
        This context is convenient for writing UDD sequence for example.
    """
    context = AlignFunc(duration=duration, func=func)
    return context.align(schedule)
