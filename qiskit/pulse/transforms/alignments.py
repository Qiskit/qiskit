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
from typing import Callable, Dict, Any, Union

import numpy as np

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ScheduleComponent
from qiskit.pulse.utils import instruction_duration_validation, deprecated_functionality


class AlignmentKind(abc.ABC):
    """An abstract class for schedule alignment."""

    is_sequential = None

    def __init__(self):
        """Create new context."""
        self._context_params = tuple()

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


class AlignLeft(AlignmentKind):
    """Align instructions in as-soon-as-possible manner.

    Instructions are placed at earliest available timeslots.
    """

    is_sequential = False

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


class AlignRight(AlignmentKind):
    """Align instructions in as-late-as-possible manner.

    Instructions are placed at latest available timeslots.
    """

    is_sequential = False

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
    def _push_right_prepend(this: ScheduleComponent, other: ScheduleComponent) -> Schedule:
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


class AlignSequential(AlignmentKind):
    """Align instructions sequentially.

    Instructions played on different channels are also arranged in a sequence.
    No buffer time is inserted in between instructions.
    """

    is_sequential = True

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
                If the specified duration is shorter than the schedule duration,
                no alignment is performed and the input schedule is just returned.
                This duration can be parametrized.
        """
        super().__init__()

        self._context_params = (duration,)

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
                If the specified duration is shorter than the schedule duration,
                no alignment is performed and the input schedule is just returned.
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
            _t_center = self.duration * self._func(ind + 1)
            _t0 = int(_t_center - 0.5 * child.duration)
            if _t0 < 0 or _t0 > self.duration:
                PulseError("Invalid schedule position t=%d is specified at index=%d" % (_t0, ind))
            aligned.insert(_t0, child, inplace=True)

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
