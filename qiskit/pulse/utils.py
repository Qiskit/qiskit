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
Pulse utilities.
"""
import warnings

from typing import List

from .channels import AcquireChannel, MemorySlot
from .commands import Acquire, AcquireInstruction
from .interfaces import ScheduleComponent
from .schedule import Schedule
from .cmd_def import CmdDef


def align_measures(schedule: ScheduleComponent, cmd_def: CmdDef) -> Schedule:
    """Return a new schedule where measurements occur at the same physical time, with the remaining
    schedules appropriately offset. Minimum measurement wait time (to allow for calibration pulses)
    is enforced.

    Args:
        schedule: Schedule to be aligned
        cmd_def: Command definition list
    Returns:
        Schedule
    Raises:
        ValueError: if an acquire or pulse is encountered on a channel that has already been part
                    of an acquire
    """
    new_schedule = Schedule()

    # Need time to allow for calibration pulses to be played for result classification
    max_calibration_duration = 0
    for qubits in cmd_def.cmd_qubits('u2'):
        cmd = cmd_def.get('u2', qubits)
        max_calibration_duration = max(cmd.duration, max_calibration_duration)

    # Schedule the acquires to be either at the end of the needed calibration time, or when the
    # last acquire is scheduled, whichever comes later
    last_acquire = max([time for time, inst in schedule.instructions
                        if isinstance(inst, AcquireInstruction)])
    acquire_scheduled_time = max(max_calibration_duration, last_acquire)

    # Shift acquires according to the new scheduled time
    acquired_channels = set()
    for time, inst in schedule.instructions:
        for chan in inst.channels:
            if chan.index in acquired_channels:
                raise ValueError("Pulse encountered on channel {0} after acquire on "
                                 "same channel.".format(chan.index))
        if isinstance(inst, AcquireInstruction):
            new_schedule |= inst << acquire_scheduled_time
            acquired_channels.update({a.index for a in inst.acquires})
        else:
            new_schedule |= inst << time

    return new_schedule


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
            new_schedule |= AcquireInstruction(
                cmd,
                [AcquireChannel(i) for i in all_qubits],
                [MemorySlot(i) for i in all_qubits]) << time
        else:
            new_schedule |= inst << time

    return new_schedule
