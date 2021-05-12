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

from qiskit.pulse import instructions
from qiskit.pulse.channels import Channel, PulseChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Interval, Schedule, ScheduleBlock
from qiskit.pulse.transforms import inline_subroutines
from qiskit.pulse.utils import format_parameter_value


def filter_instructions(
    sched: Union[Schedule, ScheduleBlock],
    filters: Union[Callable, List[Callable]],
    negate: bool = False,
    recurse_subroutines: bool = True,
) -> Schedule:
    """A filtering function that takes a schedule and returns a schedule consisting of
    filtered instructions.

    Args:
        sched: A pulse program to be filtered.
        filters: List of callback functions that take an instruction and return boolean.
        negate: Set `True` to accept an instruction if a filter function returns `False`.
            Otherwise the instruction is accepted when the filter function returns `False`.
        recurse_subroutines: Set `True` to individually filter instructions inside of a subroutine
            defined by the :py:class:`~qiskit.pulse.instructions.Call` instruction.
            When this option is enabled, call instructions are inlined and thus the context of
            subroutine will be lost.

    Returns:
        Filtered pulse schedule.
    """
    try:
        filters = list(filters)
    except TypeError:
        filters = [filters]

    filtered_sched = sched.__class__.initialize_from(sched)

    if recurse_subroutines:
        sched = inline_subroutines(sched)

    for node in sched.nodes():
        is_valid = all(filt((node.time, node.data)) for filt in filters)
        if (not is_valid and not negate) or (is_valid and negate):
            # node matches with the filter condition
            pulse_chans = [chan for chan in node.data.channels if isinstance(chan, PulseChannel)]
            node_dur = format_parameter_value(node.data.duration)
            if node_dur != 0 and pulse_chans:
                # insert placeholder.
                # note that removing node from a block may change following node scheduling.
                for chan in pulse_chans:
                    filtered_sched.add_node(instructions.Delay(node_dur, chan), node.time)
            else:
                # remove node
                continue
        else:
            if isinstance(node.data, (Schedule, ScheduleBlock)):
                filtered_sched.add_node(
                    filter_instructions(node.data, filters, negate, recurse_subroutines), node.time
                )
            else:
                filtered_sched.add_node(node.data, node.time)

    return filtered_sched


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
        time, data = time_inst
        for t0, t1 in ranges:
            if time is None:
                raise PulseError(
                    f"Instruction {data.name} is not scheduled. "
                    "This schedule cannot be filtered by intervals."
                )
            inst_start = time
            inst_stop = inst_start + data.duration
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
