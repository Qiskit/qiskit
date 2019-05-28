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
Schedule utilities.
"""
import warnings

from typing import List

from qiskit.providers.models import Command
from qiskit.qobj import PulseLibraryItem

from .channels import AcquireChannel, MemorySlot
from .commands import Acquire, AcquireInstruction
from .interfaces import ScheduleComponent
from .schedule import Schedule


def align_measures(schedule: ScheduleComponent, cmd_def: List[Command],
                   pulse_library: List[PulseLibraryItem]) -> Schedule:
    """Return a new schedule where measurements occur at the same physical time, with the remaining
    schedules appropriately offset. Minimum measurement wait time (to allow for calibration pulses)
    is enforced.

    Args:
        schedule: Schedule to be aligned
        cmd_def: Command definition list
        pulse_library: A list of pulse definitions
    """
    new_schedule = Schedule()

    pulse_library = {p.name: p.samples for p in pulse_library}

    # Need time to allow for calibration pulses to be played for result classification
    max_calibration_duration = 0
    for cmd in cmd_def:
        if cmd.name == 'u2':
            duration = sum([len(pulse_library[pulse.name]) for pulse in cmd.sequence
                            if pulse.name in pulse_library])
            max_calibration_duration = max(duration, max_calibration_duration)

    last_acquire = max([time for time, inst in schedule.instructions
                        if isinstance(inst, AcquireInstruction)])
    # Schedule the acquires to be either at the end of the needed calibration time, or when the
    # last acquire is scheduled, whichever comes later
    acquire_scheduled_time = max(max_calibration_duration, last_acquire)

    # Shift instruction times according to the new scheduled time for the acquires
    extra_delays = {}
    for time, inst in schedule.instructions:
        if isinstance(inst, AcquireInstruction):
            new_schedule |= inst << acquire_scheduled_time
            indices = [a.index for a in inst.acquires]
            extra_delays.update({i: acquire_scheduled_time - time for i in indices})
        else:
            delay = max([extra_delays.get(c.index, 0) for c in inst.timeslots.channels])
            # TODO: increase the delay on the non max channels?
            new_schedule |= inst << time + delay

    return new_schedule


def replace_implicit_measures(schedule: ScheduleComponent, meas_map: List[List[int]]) -> Schedule:
    """Return a new schedule with implicit acquires from the measurement mapping replaced by
    explicit ones.

    Warning:
        Since new acquires are being added, Memory Slots will be set to match the qubit index. This
        may overwrite your specification.

    Args:
        schedule: Schedule to be aligned
        meas_map: List of lists of qubits that are measured together
    """
    new_schedule = Schedule(name=schedule.name)

    for time, inst in schedule.instructions:
        if isinstance(inst, AcquireInstruction):
            # Get the label of all qubits that are measured with the qubit(s) in this instruction
            meas_group = [g for g in meas_map if inst.acquires[0].index in g][0]
            if any([acq.index != mem.index for acq, mem in zip(inst.acquires, inst.mem_slots)]):
                warnings.warn("One of your acquires was mapped to a memory slot which didn't match"
                              " the qubit index. I'm relabeling them to match.")
            cmd = Acquire(inst.duration, inst.command.discriminator, inst.command.kernel)
            # Replace the old acquire instruction by a new one explicitly acquiring all qubits in
            # the measurement group.
            new_schedule |= AcquireInstruction(
                cmd,
                [AcquireChannel(i) for i in meas_group],
                [MemorySlot(i) for i in meas_group]) << time
        else:
            new_schedule |= inst << time

    return new_schedule
