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

"""
Basic rescheduling functions which take schedules or instructions
(and possibly some arguments) and return new schedules.
"""
import warnings

import collections
from typing import List, Optional, Iterable

import numpy as np

import qiskit.pulse.channels as chans
import qiskit.pulse.commands as commands
import qiskit.pulse.exceptions as exceptions
import qiskit.pulse.instructions as instructions
import qiskit.pulse.instructions.directives as directives
import qiskit.pulse.interfaces as interfaces
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
                             if isinstance(inst, (instructions.Acquire,
                                                  commands.AcquireInstruction))]
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

            if isinstance(inst, (instructions.Acquire, commands.AcquireInstruction)):
                if time > align_time:
                    warnings.warn("You provided an align_time which is scheduling an acquire "
                                  "sooner than it was scheduled for in the original Schedule.")
                new_schedule |= inst << align_time
                acquired_channels.update({a.index for a in inst.acquires})
            elif isinstance(inst.channels[0], chans.MeasureChannel):
                new_schedule |= inst << align_time
                measured_channels.update({a.index for a in inst.channels})
            else:
                new_schedule |= inst << time

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
        A ``Schedule`` with the additional acquisition commands.
    """
    new_schedule = Schedule(name=schedule.name)
    acquire_map = dict()

    for time, inst in schedule.instructions:
        if isinstance(inst, (instructions.Acquire, commands.AcquireInstruction)):
            if any([acq.index != mem.index for acq, mem in zip(inst.acquires, inst.mem_slots)]):
                warnings.warn("One of your acquires was mapped to a memory slot which didn't match"
                              " the qubit index. I'm relabeling them to match.")

            # Get the label of all qubits that are measured with the qubit(s) in this instruction
            existing_qubits = {chan.index for chan in inst.acquires}
            all_qubits = []
            for sublist in meas_map:
                if existing_qubits.intersection(set(sublist)):
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
            new_schedule |= inst << time

    return new_schedule


def pad(schedule: Schedule,
        channels: Optional[Iterable[chans.Channel]] = None,
        until: Optional[int] = None,
        mutate: bool = False
        ) -> Schedule:
    r"""Pad the input Schedule with ``Delay``\s on all unoccupied timeslots until
    ``schedule.duration`` or ``until`` if not ``None``.

    Args:
        schedule: Schedule to pad.
        channels: Channels to pad. Defaults to all channels in
            ``schedule`` if not provided. If the supplied channel is not a member
            of ``schedule`` it will be added.
        until: Time to pad until. Defaults to ``schedule.duration`` if not provided.
        mutate: Pad this schedule by mutating rather than returning a new schedule.

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
                    mutate=mutate)
            curr_time = interval[1]
        if curr_time < until:
            schedule = schedule.insert(
                curr_time,
                instructions.Delay(until - curr_time, channel),
                mutate=mutate)

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
                    new_schedule |= inst << time
            else:
                new_schedule |= inst << time

        new_schedules.append(new_schedule)

    return new_schedules


def push_append(this: List[interfaces.ScheduleComponent],
                other: List[interfaces.ScheduleComponent],
                ) -> Schedule:
    r"""Return a ``this`` with ``other`` inserted at the maximum time over
    all channels shared between ```this`` and ``other``.

   $t = \textrm{max}({x.stop\_time |x \in self.channels \cap schedule.channels})$

    Args:
        this: Input schedule to which ``other`` will be inserted.
        other: Other schedule to insert.

    Returns:
        Joined schedule
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

    other_only_insert_time = other.ch_start_time(*(other_channels - this_channels))
    insert_time = max(shared_insert_time, other_only_insert_time)
    a = this.insert(insert_time, other, mutate=True)
    return a


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
    for _, child in schedule.children:
        push_append(aligned, child)

    return aligned


def align_right(schedule: Schedule) -> Schedule:
    """Align a list of pulse instructions on the right.

    Args:
        schedule: Input schedule of which top-level ``child`` nodes will be
            reschedulued.

    Returns:
        New schedule with input `schedule`` child schedules and instructions
        right aligned.
    """
    left_aligned = align_left(schedule)
    total_duration = left_aligned.duration

    latest_available_times = collections.defaultdict(lambda: total_duration)
    right_aligned = Schedule()
    for _, child in reversed(left_aligned.children):
        child_channels = child.channels
        # drop node without channels as it is empty
        if child_channels:
            latest_available_duration = min(latest_available_times[channel] for
                                            channel in child_channels)
            insert_time = latest_available_duration - child.duration
            right_aligned.insert(insert_time, child, mutate=True)
            latest_available_times.update(
                dict.fromkeys(child_channels, insert_time))

    return right_aligned


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
    for _, child in schedule.children:
        aligned.insert(aligned.duration, child, mutate=True)
    return aligned


def barrier_left(schedule: Schedule,
                 channels: Optional[Iterable[chans.Channel]] = None
                 ) -> Schedule:
    """Align on the left and create a barrier so that pulses cannot be inserted
        within this pulse interval.

    Args:
        schedule: Input schedule of which top-level ``child`` nodes will be
            reschedulued.
        channels: Channels to barrier. Defaults to all channels in ``schedule``
            if not in supplied.

    Returns:
        New schedule with input `schedule`` child schedules and instructions
        left barriered.
    """
    aligned = align_left(schedule)
    return pad(aligned, channels=channels)


def barrier_right(schedule: Schedule,
                  channels: Optional[Iterable[chans.Channel]] = None
                  ) -> Schedule:
    """Align on the right and create a barrier so that pulses cannot be
        inserted within this pulse interval.

    Args:
        schedule: Input schedule of which top-level ``child`` nodes will be
            reschedulued.
        channels: Channels to barrier. Defaults to all channels in ``schedule``
            if not in supplied.

    Returns:
        New schedule with input `schedule`` child schedules and instructions
        right barriered.
    """
    aligned = align_left(schedule)
    return pad(aligned, channels=channels)


def group(schedule: Schedule) -> Schedule:
    """Group all instructions in a node.

    With the current schedule implementation this is trivial.
    """
    return schedule


def flatten(schedule: Schedule) -> Schedule:
    """Flatten any grouped nodes."""
    return schedule.flatten()


def remove_directives(schedule: Schedule) -> Schedule:
    """Remove directives."""
    return schedule.exclude(instruction_types=[directives.Directive])
