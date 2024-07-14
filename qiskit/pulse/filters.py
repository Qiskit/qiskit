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
from __future__ import annotations
import abc
from functools import singledispatch
from collections.abc import Iterable
from typing import Callable, Any, List

import numpy as np

from qiskit.pulse import Schedule, ScheduleBlock, Instruction
from qiskit.pulse.channels import Channel
from qiskit.pulse.schedule import Interval
from qiskit.pulse.exceptions import PulseError


@singledispatch
def filter_instructions(
    sched,
    filters: List[Callable[..., bool]],
    negate: bool = False,
    recurse_subroutines: bool = True,
):
    """A catch-TypeError function which will only get called if none of the other decorated
    functions, namely handle_schedule() and handle_scheduleblock(), handle the type passed.
    """
    raise TypeError(
        f"Type '{type(sched)}' is not valid data format as an input to filter_instructions."
    )


@filter_instructions.register
def handle_schedule(
    sched: Schedule,
    filters: List[Callable[..., bool]],
    negate: bool = False,
    recurse_subroutines: bool = True,
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


@filter_instructions.register
def handle_scheduleblock(
    sched_blk: ScheduleBlock,
    filters: List[Callable[..., bool]],
    negate: bool = False,
    recurse_subroutines: bool = True,
) -> ScheduleBlock:
    """A filtering function that takes a schedule_block and returns a schedule_block consisting of
    filtered instructions.

    Args:
        sched_blk: A pulse schedule_block to be filtered.
        filters: List of callback functions that take an instruction and return boolean.
        negate: Set `True` to accept an instruction if a filter function returns `False`.
            Otherwise the instruction is accepted when the filter function returns `False`.
        recurse_subroutines: Set `True` to individually filter instructions inside of a subroutine
            defined by the :py:class:`~qiskit.pulse.instructions.Call` instruction.

    Returns:
        Filtered pulse schedule_block.
    """
    from qiskit.pulse.transforms import inline_subroutines

    target_sched_blk = sched_blk
    if recurse_subroutines:
        target_sched_blk = inline_subroutines(target_sched_blk)

    def apply_filters_to_insts_in_scheblk(blk: ScheduleBlock) -> ScheduleBlock:
        blk_new = ScheduleBlock.initialize_from(blk)
        for element in blk.blocks:
            if isinstance(element, ScheduleBlock):
                inner_blk = apply_filters_to_insts_in_scheblk(element)
                if len(inner_blk) > 0:
                    blk_new.append(inner_blk)

            elif isinstance(element, Instruction):
                valid_inst = all(filt(element) for filt in filters)
                if negate:
                    valid_inst ^= True
                if valid_inst:
                    blk_new.append(element)

            else:
                raise PulseError(
                    f"An unexpected element '{element}' is included in ScheduleBlock.blocks."
                )
        return blk_new

    filter_sched_blk = apply_filters_to_insts_in_scheblk(target_sched_blk)
    return filter_sched_blk


def composite_filter(
    channels: Iterable[Channel] | Channel | None = None,
    instruction_types: Iterable[abc.ABCMeta] | abc.ABCMeta | None = None,
    time_ranges: Iterable[tuple[int, int]] | None = None,
    intervals: Iterable[Interval] | None = None,
) -> list[Callable]:
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


def with_channels(channels: Iterable[Channel] | Channel) -> Callable:
    """Channel filter generator.

    Args:
        channels: List of channels to filter.

    Returns:
        A callback function to filter channels.
    """
    channels = _if_scalar_cast_to_list(channels)

    @singledispatch
    def channel_filter(time_inst):
        """A catch-TypeError function which will only get called if none of the other decorated
        functions, namely handle_numpyndarray() and handle_instruction(), handle the type passed.
        """
        raise TypeError(
            f"Type '{type(time_inst)}' is not valid data format as an input to channel_filter."
        )

    @channel_filter.register
    def handle_numpyndarray(time_inst: np.ndarray) -> bool:
        """Filter channel.

        Args:
            time_inst (numpy.ndarray([int, Instruction])): Time

        Returns:
            If instruction matches with condition.
        """
        return any(chan in channels for chan in time_inst[1].channels)

    @channel_filter.register
    def handle_instruction(inst: Instruction) -> bool:
        """Filter channel.

        Args:
            inst: Instruction

        Returns:
            If instruction matches with condition.
        """
        return any(chan in channels for chan in inst.channels)

    return channel_filter


def with_instruction_types(types: Iterable[abc.ABCMeta] | abc.ABCMeta) -> Callable:
    """Instruction type filter generator.

    Args:
        types: List of instruction types to filter.

    Returns:
        A callback function to filter instructions.
    """
    types = _if_scalar_cast_to_list(types)

    @singledispatch
    def instruction_filter(time_inst) -> bool:
        """A catch-TypeError function which will only get called if none of the other decorated
        functions, namely handle_numpyndarray() and handle_instruction(), handle the type passed.
        """
        raise TypeError(
            f"Type '{type(time_inst)}' is not valid data format as an input to instruction_filter."
        )

    @instruction_filter.register
    def handle_numpyndarray(time_inst: np.ndarray) -> bool:
        """Filter instruction.

        Args:
            time_inst (numpy.ndarray([int, Instruction])): Time

        Returns:
            If instruction matches with condition.
        """
        return isinstance(time_inst[1], tuple(types))

    @instruction_filter.register
    def handle_instruction(inst: Instruction) -> bool:
        """Filter instruction.

        Args:
            inst: Instruction

        Returns:
            If instruction matches with condition.
        """
        return isinstance(inst, tuple(types))

    return instruction_filter


def with_intervals(ranges: Iterable[Interval] | Interval) -> Callable:
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


def _if_scalar_cast_to_list(to_list: Any) -> list[Any]:
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
