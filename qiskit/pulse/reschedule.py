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
Basic rescheduling functions which take a Schedule (and possibly some arguments) and return
a new Schedule.
"""
import warnings

from typing import List, Optional, Iterable

import numpy as np

from .channels import Channel, AcquireChannel, MeasureChannel, MemorySlot
from .cmd_def import CmdDef
from .commands import Acquire, AcquireInstruction, Delay
from .exceptions import PulseError
from .instruction_schedule_map import InstructionScheduleMap
from .interfaces import ScheduleComponent
from .schedule import Schedule


def align_measures(schedules: Iterable[ScheduleComponent],
                   inst_map: Optional[InstructionScheduleMap] = None,
                   cmd_def: Optional[CmdDef] = None,
                   cal_gate: str = 'u3',
                   max_calibration_duration: Optional[int] = None,
                   align_time: Optional[int] = None) -> Schedule:
    """Return new schedules where measurements occur at the same physical time. Minimum measurement
    wait time (to allow for calibration pulses) is enforced.

    This is only defined for schedules that are acquire-less or acquire-final per channel: a
    schedule with pulses or acquires occurring on a channel which has already had a measurement will
    throw an error.

    Args:
        schedules: Collection of schedules to be aligned together
        inst_map: Mapping of circuit operations to pulse schedules
        cmd_def: Deprecated
        cal_gate: The name of the gate to inspect for the calibration time
        max_calibration_duration: If provided, cmd_def and cal_gate will be ignored
        align_time: If provided, this will be used as final align time.
    Returns:
        Schedule
    Raises:
        PulseError: if an acquire or pulse is encountered on a channel that has already been part
                    of an acquire, or if align_time is negative
    """
    if inst_map is None:
        inst_map = cmd_def

    def calculate_align_time():
        """Return the the max between the duration of the calibration time and the absolute time
        of the latest scheduled acquire."""
        nonlocal max_calibration_duration
        if max_calibration_duration is None:
            max_calibration_duration = get_max_calibration_duration()
        align_time = max_calibration_duration
        for schedule in schedules:
            last_acquire = 0
            acquire_times = [time for time, inst in schedule.instructions
                             if isinstance(inst, AcquireInstruction)]
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
        raise PulseError("Must provide a inst_map, an alignment time, or a calibration duration.")
    if align_time is not None and align_time < 0:
        raise PulseError("Align time cannot be negative.")
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
                if isinstance(chan, MeasureChannel):
                    if chan.index in measured_channels:
                        raise PulseError("Multiple measurements are not supported by this "
                                         "rescheduling pass.")
                elif chan.index in acquired_channels:
                    raise PulseError("Pulse encountered on channel {0} after acquire on "
                                     "same channel.".format(chan.index))

            if isinstance(inst, AcquireInstruction):
                if time > align_time:
                    warnings.warn("You provided an align_time which is scheduling an acquire "
                                  "sooner than it was scheduled for in the original Schedule.")
                new_schedule |= inst << align_time
                acquired_channels.update({a.index for a in inst.acquires})
            elif isinstance(inst.channels[0], MeasureChannel):
                new_schedule |= inst << align_time
                measured_channels.update({a.index for a in inst.channels})
            else:
                new_schedule |= inst << time

        new_schedules.append(new_schedule)

    return new_schedules


def add_implicit_acquires(schedule: ScheduleComponent, meas_map: List[List[int]]) -> Schedule:
    """Return a new schedule with implicit acquires from the measurement mapping replaced by
    explicit ones.

    Warning:
        Since new acquires are being added, Memory Slots will be set to match the qubit index. This
        may overwrite your specification.

    Args:
        schedule: Schedule to be aligned
        meas_map: List of lists of qubits that are measured together
    Returns:
        Schedule
    """
    new_schedule = Schedule(name=schedule.name)
    acquire_map = dict()

    for time, inst in schedule.instructions:
        if isinstance(inst, AcquireInstruction):
            if any([acq.index != mem.index for acq, mem in zip(inst.acquires, inst.mem_slots)]):
                warnings.warn("One of your acquires was mapped to a memory slot which didn't match"
                              " the qubit index. I'm relabeling them to match.")

            cmd = Acquire(inst.duration, inst.command.discriminator, inst.command.kernel)
            # Get the label of all qubits that are measured with the qubit(s) in this instruction
            existing_qubits = {chan.index for chan in inst.acquires}
            all_qubits = []
            for sublist in meas_map:
                if existing_qubits.intersection(set(sublist)):
                    all_qubits.extend(sublist)
            # Replace the old acquire instruction by a new one explicitly acquiring all qubits in
            # the measurement group.
            for i in all_qubits:
                explicit_inst = AcquireInstruction(cmd, AcquireChannel(i), MemorySlot(i)) << time
                if time not in acquire_map:
                    new_schedule |= explicit_inst
                    acquire_map = {time: {i}}
                elif i not in acquire_map[time]:
                    new_schedule |= explicit_inst
                    acquire_map[time].add(i)
        else:
            new_schedule |= inst << time

    return new_schedule


def pad(schedule: Schedule, channels: Optional[Iterable[Channel]] = None,
        until: Optional[int] = None) -> Schedule:
    """Pad the input Schedule with `Delay`s on all unoccupied timeslots until
    `schedule.duration` or `until` if not `None`.

    Args:
        schedule: Schedule to pad.
        channels: Channels to pad. Defaults to all channels in
            `schedule` if not provided. If the supplied channel is not a member
            of `schedule` it will be added.
        until: Time to pad until. Defaults to `schedule.duration` if not provided.
    Returns:
        Schedule: The padded schedule
    """
    until = until or schedule.duration

    channels = channels or schedule.channels
    occupied_channels = schedule.channels

    unoccupied_channels = set(channels) - set(occupied_channels)

    empty_timeslot_collection = schedule.timeslots.complement(until)

    for channel in channels:
        for timeslot in empty_timeslot_collection.ch_timeslots(channel):
            schedule |= Delay(timeslot.duration)(timeslot.channel).shift(timeslot.start)

    for channel in unoccupied_channels:
        schedule |= Delay(until)(channel)

    return schedule
