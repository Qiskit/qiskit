# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Basic rescheduling functions which take schedules or instructions
(and possibly some arguments) and return new schedules.
"""
import warnings

from typing import List, Optional, Iterable

import numpy as np

from qiskit.pulse import channels as chans, exceptions, instructions, interfaces
from qiskit.pulse.instructions import directives
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.schedule import Schedule


def align_measures(schedules: Iterable[interfaces.ScheduleComponent],
                   inst_map: Optional[InstructionScheduleMap] = None,
                   cal_gate: str = 'u3',
                   max_calibration_duration: Optional[int] = None,
                   align_time: Optional[int] = None
                   ) -> List[Schedule]:
    """Return new schedules where measurements occur at the same physical time. Minimum measurement
    wait time (to allow for calibration pulses) is enforced.

    This is only defined for schedules that are acquire-less or acquire-final per channel: a
    schedule with pulses or acquires occurring on a channel which has already had a measurement will
    throw an error.

    Args:
        schedules: Collection of schedules to be aligned together
        inst_map: Mapping of circuit operations to pulse schedules
        cal_gate: The name of the gate to inspect for the calibration time
        max_calibration_duration: If provided, inst_map and cal_gate will be ignored
        align_time: If provided, this will be used as final align time.

    Returns:
        The input list of schedules transformed to have their measurements aligned.

    Raises:
        PulseError: if an acquire or pulse is encountered on a channel that has already been part
                    of an acquire, or if align_time is negative
    """
    def calculate_align_time():
        """Return the the max between the duration of the calibration time and the absolute time
        of the latest scheduled acquire.
        """
        nonlocal max_calibration_duration
        if max_calibration_duration is None:
            max_calibration_duration = get_max_calibration_duration()
        align_time = max_calibration_duration
        for schedule in schedules:
            last_acquire = 0
            acquire_times = [time for time, inst in schedule.instructions
                             if isinstance(inst, instructions.Acquire)]
            if acquire_times:
                last_acquire = max(acquire_times)
            align_time = max(align_time, last_acquire)
        return align_time

    def get_max_calibration_duration():
        """Return the time needed to allow for readout discrimination calibration pulses."""
        max_calibration_duration = 0
        for qubits in inst_map.qubits_with_instruction(cal_gate):
            cmd = inst_map.get(cal_gate, qubits, np.pi, 0, np.pi)
            max_calibration_duration = max(cmd.duration, max_calibration_duration)
        return max_calibration_duration

    if align_time is None and max_calibration_duration is None and inst_map is None:
        raise exceptions.PulseError("Must provide a inst_map, an alignment time, "
                                    "or a calibration duration.")
    if align_time is not None and align_time < 0:
        raise exceptions.PulseError("Align time cannot be negative.")
    if align_time is None:
        align_time = calculate_align_time()

    # Shift acquires according to the new scheduled time
    new_schedules = []
    for schedule in schedules:
        new_schedule = Schedule(name=schedule.name)
        acquired_channels = set()
        measured_channels = set()

        for time, inst in schedule.instructions:
            for chan in inst.channels:
                if isinstance(chan, chans.MeasureChannel):
                    if chan.index in measured_channels:
                        raise exceptions.PulseError("Multiple measurements are "
                                                    "not supported by this "
                                                    "rescheduling pass.")
                elif chan.index in acquired_channels:
                    raise exceptions.PulseError("Pulse encountered on channel "
                                                "{0} after acquire on "
                                                "same channel.".format(chan.index))

            if isinstance(inst, instructions.Acquire):
                if time > align_time:
                    warnings.warn("You provided an align_time which is scheduling an acquire "
                                  "sooner than it was scheduled for in the original Schedule.")
                new_schedule.insert(align_time, inst, inplace=True)
                acquired_channels.add(inst.channel.index)
            elif isinstance(inst.channels[0], chans.MeasureChannel):
                new_schedule.insert(align_time, inst, inplace=True)
                measured_channels.update({a.index for a in inst.channels})
            else:
                new_schedule.insert(time, inst, inplace=True)

        new_schedules.append(new_schedule)

    return new_schedules


def add_implicit_acquires(schedule: interfaces.ScheduleComponent,
                          meas_map: List[List[int]]
                          ) -> Schedule:
    """Return a new schedule with implicit acquires from the measurement mapping replaced by
    explicit ones.

    .. warning:: Since new acquires are being added, Memory Slots will be set to match the
                 qubit index. This may overwrite your specification.

    Args:
        schedule: Schedule to be aligned.
        meas_map: List of lists of qubits that are measured together.

    Returns:
        A ``Schedule`` with the additional acquisition instructions.
    """
    new_schedule = Schedule(name=schedule.name)
    acquire_map = dict()

    for time, inst in schedule.instructions:
        if isinstance(inst, instructions.Acquire):
            if inst.mem_slot and inst.mem_slot.index != inst.channel.index:
                warnings.warn("One of your acquires was mapped to a memory slot which didn't match"
                              " the qubit index. I'm relabeling them to match.")

            # Get the label of all qubits that are measured with the qubit(s) in this instruction
            all_qubits = []
            for sublist in meas_map:
                if inst.channel.index in sublist:
                    all_qubits.extend(sublist)
            # Replace the old acquire instruction by a new one explicitly acquiring all qubits in
            # the measurement group.
            for i in all_qubits:
                explicit_inst = instructions.Acquire(inst.duration,
                                                     chans.AcquireChannel(i),
                                                     mem_slot=chans.MemorySlot(i),
                                                     kernel=inst.kernel,
                                                     discriminator=inst.discriminator) << time
                if time not in acquire_map:
                    new_schedule |= explicit_inst
                    acquire_map = {time: {i}}
                elif i not in acquire_map[time]:
                    new_schedule |= explicit_inst
                    acquire_map[time].add(i)
        else:
            new_schedule.insert(time, inst, inplace=True)

    return new_schedule


def pad(schedule: Schedule,
        channels: Optional[Iterable[chans.Channel]] = None,
        until: Optional[int] = None,
        inplace: bool = False
        ) -> Schedule:
    r"""Pad the input Schedule with ``Delay``\s on all unoccupied timeslots until
    ``schedule.duration`` or ``until`` if not ``None``.

    Args:
        schedule: Schedule to pad.
        channels: Channels to pad. Defaults to all channels in
            ``schedule`` if not provided. If the supplied channel is not a member
            of ``schedule`` it will be added.
        until: Time to pad until. Defaults to ``schedule.duration`` if not provided.
        inplace: Pad this schedule by mutating rather than returning a new schedule.

    Returns:
        The padded schedule.
    """
    until = until or schedule.duration
    channels = channels or schedule.channels

    for channel in channels:
        if channel not in schedule.channels:
            schedule |= instructions.Delay(until, channel)
            continue

        curr_time = 0
        # TODO: Replace with method of getting instructions on a channel
        for interval in schedule.timeslots[channel]:
            if curr_time >= until:
                break
            if interval[0] != curr_time:
                end_time = min(interval[0], until)
                schedule = schedule.insert(
                    curr_time,
                    instructions.Delay(end_time - curr_time, channel),
                    inplace=inplace)
            curr_time = interval[1]
        if curr_time < until:
            schedule = schedule.insert(
                curr_time,
                instructions.Delay(until - curr_time, channel),
                inplace=inplace)

    return schedule


def compress_pulses(schedules: List[Schedule]) -> List[Schedule]:
    """Optimization pass to replace identical pulses.

    Args:
        schedules: Schedules to compress.

    Returns:
        Compressed schedules.
    """

    existing_pulses = []
    new_schedules = []

    for schedule in schedules:
        new_schedule = Schedule(name=schedule.name)

        for time, inst in schedule.instructions:
            if isinstance(inst, instructions.Play):
                if inst.pulse in existing_pulses:
                    idx = existing_pulses.index(inst.pulse)
                    identical_pulse = existing_pulses[idx]
                    new_schedule |= instructions.Play(
                        identical_pulse, inst.channel, inst.name) << time
                else:
                    existing_pulses.append(inst.pulse)
                    new_schedule.insert(time, inst, inplace=True)
            else:
                new_schedule.insert(time, inst, inplace=True)

        new_schedules.append(new_schedule)

    return new_schedules


def _push_left_append(this: Schedule,
                      other: interfaces.ScheduleComponent,
                      ) -> Schedule:
    r"""Return ``this`` with ``other`` inserted at the maximum time over
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
    ch_slacks = [this.stop_time - this.ch_stop_time(channel) + other.ch_start_time(channel)
                 for channel in shared_channels]

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


def align_left(schedule: Schedule) -> Schedule:
    """Align a list of pulse instructions on the left.

    Args:
        schedule: Input schedule of which top-level ``child`` nodes will be
            reschedulued.

    Returns:
        New schedule with input `schedule`` child schedules and instructions
        left aligned.
    """
    aligned = Schedule()
    for _, child in schedule._children:
        _push_left_append(aligned, child)
    return aligned


def _push_right_prepend(this: interfaces.ScheduleComponent,
                        other: interfaces.ScheduleComponent,
                        ) -> Schedule:
    r"""Return ``this`` with ``other`` inserted at the latest possible time
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
    ch_slacks = [this.ch_start_time(channel) - other.ch_stop_time(channel)
                 for channel in shared_channels]

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


def align_right(schedule: Schedule) -> Schedule:
    """Align a list of pulse instructions on the right.

    Args:
        schedule: Input schedule of which top-level ``child`` nodes will be
            reschedulued.

    Returns:
        New schedule with input `schedule`` child schedules and instructions
        right aligned.
    """
    aligned = Schedule()
    for _, child in reversed(schedule._children):
        aligned = _push_right_prepend(aligned, child)
    return aligned


def align_sequential(schedule: Schedule) -> Schedule:
    """Schedule all top-level nodes in parallel.

    Args:
        schedule: Input schedule of which top-level ``child`` nodes will be
            reschedulued.

    Returns:
        New schedule with input `schedule`` child schedules and instructions
        applied sequentially across channels
    """
    aligned = Schedule()
    for _, child in schedule._children:
        aligned.insert(aligned.duration, child, inplace=True)
    return aligned


def flatten(schedule: Schedule) -> Schedule:
    """Flatten any called nodes into a Schedule tree with no nested children."""
    return schedule.flatten()


def remove_directives(schedule: Schedule) -> Schedule:
    """Remove directives."""
    return schedule.exclude(instruction_types=[directives.Directive])


def remove_trivial_barriers(schedule: Schedule) -> Schedule:
    """Remove trivial barriers with 0 or 1 channels."""
    def filter_func(inst):
        return (isinstance(inst[1], directives.RelativeBarrier) and
                len(inst[1].channels) < 2)

    return schedule.exclude(filter_func)
