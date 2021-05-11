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

"""A collection of functions that filter instructions in a pulse program."""

import abc
from typing import Callable, List, Union, Iterable, Optional, Tuple, Any

import numpy as np

from qiskit.pulse import Schedule
from qiskit.pulse.channels import Channel
from qiskit.pulse.schedule import Interval


def filter_instructions(
    sched: Schedule, filters: List[Callable], negate: bool = False, recurse_subroutines: bool = True
) -> Schedule:
    """A filtering function that takes a schedule and returns a schedule consisting of
    filtered instructions.

    Args:
        sched: A pulse schedule to be filtered.
        filters: List of callback functions that take an instruction and return boolean.
        negate: Set `True` to accept an instruction if a filter function returns `False`.
            Otherwise the instruction is accepted when the filter function returns `False`.
        recurse_subroutines: Set `True` to individually filter instructions inside of a subroutine
            defined by the :py:class:`~qiskit.pulse.instructions.Call` instruction.

    Returns:
        Filtered pulse schedule.
    """
    from qiskit.pulse.transforms import flatten, inline_subroutines

    target_sched = flatten(sched)
    if recurse_subroutines:
        target_sched = inline_subroutines(target_sched)

    time_inst_tuples = np.array(target_sched.instructions)

    valid_insts = np.ones(len(time_inst_tuples), dtype=bool)
    for filt in filters:
        valid_insts = np.logical_and(valid_insts, np.array(list(map(filt, time_inst_tuples))))

    if negate and len(filters) > 0:
        valid_insts = ~valid_insts

    filter_schedule = Schedule.initialize_from(sched)
    for time, inst in time_inst_tuples[valid_insts]:
        filter_schedule.insert(time, inst, inplace=True)

    return filter_schedule


def composite_filter(
    channels: Optional[Union[Iterable[Channel], Channel]] = None,
    instruction_types: Optional[Union[Iterable[abc.ABCMeta], abc.ABCMeta]] = None,
    time_ranges: Optional[Iterable[Tuple[int, int]]] = None,
    intervals: Optional[Iterable[Interval]] = None,
) -> List[Callable]:
    """A helper function to generate a list of filter functions based on
    typical elements to be filtered.

    Args:
        channels: For example, ``[DriveChannel(0), AcquireChannel(0)]``.
        instruction_types (Optional[Iterable[Type[qiskit.pulse.Instruction]]]): For example,
            ``[PulseInstruction, AcquireInstruction]``.
        time_ranges: For example, ``[(0, 5), (6, 10)]``.
        intervals: For example, ``[(0, 5), (6, 10)]``.

    Returns:
        List of filtering functions.
    """
    filters = []

    # An empty list is also valid input for filter generators.
    # See unittest `test.python.pulse.test_schedule.TestScheduleFilter.test_empty_filters`.
    if channels is not None:
        filters.append(with_channels(channels))
    if instruction_types is not None:
        filters.append(with_instruction_types(instruction_types))
    if time_ranges is not None:
        filters.append(with_intervals(time_ranges))
    if intervals is not None:
        filters.append(with_intervals(intervals))

    return filters


def with_channels(channels: Union[Iterable[Channel], Channel]) -> Callable:
    """Channel filter generator.

    Args:
        channels: List of channels to filter.

    Returns:
        A callback function to filter channels.
    """
    channels = _if_scalar_cast_to_list(channels)

    def channel_filter(time_inst) -> bool:
        """Filter channel.

        Args:
            time_inst (Tuple[int, Instruction]): Time

        Returns:
            If instruction matches with condition.
        """
        return any(chan in channels for chan in time_inst[1].channels)

    return channel_filter


def with_instruction_types(types: Union[Iterable[abc.ABCMeta], abc.ABCMeta]) -> Callable:
    """Instruction type filter generator.

    Args:
        types: List of instruction types to filter.

    Returns:
        A callback function to filter instructions.
    """
    types = _if_scalar_cast_to_list(types)

    def instruction_filter(time_inst) -> bool:
        """Filter instruction.

        Args:
            time_inst (Tuple[int, Instruction]): Time

        Returns:
            If instruction matches with condition.
        """
        return isinstance(time_inst[1], tuple(types))

    return instruction_filter


def with_intervals(ranges: Union[Iterable[Interval], Interval]) -> Callable:
    """Interval filter generator.

    Args:
        ranges: List of intervals ``[t0, t1]`` to filter.

    Returns:
        A callback function to filter intervals.
    """
    ranges = _if_scalar_cast_to_list(ranges)

    def interval_filter(time_inst) -> bool:
        """Filter interval.
        Args:
            time_inst (Tuple[int, Instruction]): Time

        Returns:
            If instruction matches with condition.
        """
        for t0, t1 in ranges:
            inst_start = time_inst[0]
            inst_stop = inst_start + time_inst[1].duration
            if t0 <= inst_start and inst_stop <= t1:
                return True
        return False

    return interval_filter


def _if_scalar_cast_to_list(to_list: Any) -> List[Any]:
    """A helper function to create python list of input arguments.

    Args:
        to_list: Arbitrary object can be converted into a python list.

    Returns:
        Python list of input object.
    """
    try:
        iter(to_list)
    except TypeError:
        to_list = [to_list]
    return to_list
