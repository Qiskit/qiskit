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

from typing import List, Optional

import numpy as np

from .channels import AcquireChannel, MemorySlot
from .cmd_def import CmdDef
from .commands import Acquire, AcquireInstruction
from .exceptions import PulseError
from .interfaces import ScheduleComponent
from .schedule import Schedule


def align_measures(schedules: List[ScheduleComponent], cmd_def: CmdDef, cal_gate: str = 'u3',
                   max_calibration_duration: Optional[int] = None,
                   align_time: Optional[int] = None) -> Schedule:
    """Return new schedules where measurements occur at the same physical time. Minimum measurement
    wait time (to allow for calibration pulses) is enforced.
    This is only defined for schedules that are acquire-less or acquire-final per channel: a
    schedule with pulses or acquires occuring on a channel which has already had an acquire will
    throw an error.

    Args:
        schedules: Collection of schedules to be aligned together
        cmd_def: Command definition list
        cal_gate: The name of the gate to inspect for the calibration time
        max_calibration_duration: If provided, cmd_def and cal_gate will be ignored
        align_time: If provided, this will be used as final align time.
    Returns:
        Schedule
    Raises:
        PulseError: if an acquire or pulse is encountered on a channel that has already been part
                    of an acquire, or
                    if align_time is negative
    """
    if align_time is not None and align_time < 0:
        raise PulseError("Align time cannot be negative.")
    if align_time is None:
        # Need time to allow for calibration pulses to be played for result classification
        if max_calibration_duration is None:
            max_calibration_duration = 0
            for qubits in cmd_def.cmd_qubits(cal_gate):
                cmd = cmd_def.get(cal_gate, qubits, np.pi, 0, np.pi)
                max_calibration_duration = max(cmd.duration, max_calibration_duration)

        # Schedule the acquires to be either at the end of the needed calibration time, or when the
        # last acquire is scheduled, whichever comes later
        align_time = max_calibration_duration
        for schedule in schedules:
            last_acquire = max([time for time, inst in schedule.instructions
                                if isinstance(inst, AcquireInstruction)])
            align_time = max(align_time, last_acquire)

    # Shift acquires according to the new scheduled time
    new_schedules = []
    for schedule in schedules:
        new_schedule = Schedule()
        acquired_channels = set()
        for time, inst in schedule.instructions:
            for chan in inst.channels:
                if chan.index in acquired_channels:
                    raise PulseError("Pulse encountered on channel {0} after acquire on "
                                     "same channel.".format(chan.index))
            if isinstance(inst, AcquireInstruction):
                if time > align_time:
                    warnings.warn("You provided an align_time which is scheduling an acquire "
                                  "sooner than it was scheduled for in the original Schedule.")
                new_schedule |= inst << align_time
                acquired_channels.update({a.index for a in inst.acquires})
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
