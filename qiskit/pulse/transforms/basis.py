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
"""Basic rescheduling functions which take schedule or instructions and return new schedules."""

from copy import deepcopy
from typing import List

from qiskit.pulse import instructions
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.instructions import directives
from qiskit.pulse.schedule import Schedule, ScheduleBlock, PulseProgram
from qiskit.pulse.exceptions import UnassignedDurationError


def block_to_schedule(block: ScheduleBlock) -> Schedule:
    """Convert ``ScheduleBlock`` to ``Schedule``.

    Args:
        block: A ``ScheduleBlock`` to convert.

    Returns:
        Scheduled pulse program.

    Raises:
        UnassignedDurationError: When any instruction duration is not assigned.
    """
    if not block.is_schedulable():
        raise UnassignedDurationError(
            'All instruction durations should be assigned before creating `Schedule`.'
            'Please check `.parameters` to find unassigned parameter objects.')

    schedule = Schedule(name=block.name, metadata=block.metadata)
    for op_data in block.instructions:
        if isinstance(op_data, ScheduleBlock):
            context_schedule = block_to_schedule(op_data)
            schedule.append(context_schedule, inplace=True)
        else:
            schedule.append(op_data, inplace=True)

    # transform with defined policy
    return block.context_alignment.align(schedule)


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
        new_schedule = Schedule(name=schedule.name, metadata=schedule.metadata)

        for time, inst in schedule.instructions:
            if isinstance(inst, instructions.Play):
                if inst.pulse in existing_pulses:
                    idx = existing_pulses.index(inst.pulse)
                    identical_pulse = existing_pulses[idx]
                    new_schedule.insert(time,
                                        instructions.Play(identical_pulse,
                                                          inst.channel,
                                                          inst.name),
                                        inplace=True)
                else:
                    existing_pulses.append(inst.pulse)
                    new_schedule.insert(time, inst, inplace=True)
            else:
                new_schedule.insert(time, inst, inplace=True)

        new_schedules.append(new_schedule)

    return new_schedules


def flatten(program: PulseProgram) -> PulseProgram:
    """Flatten (inline) any called nodes into a Schedule tree with no nested children.

    Args:
        program: Pulse program to remove nested structure.

    Returns:
        Flatten pulse program.

    Raises:
        PulseError: When invalid data format is given.
    """
    if isinstance(program, ScheduleBlock):
        return program
    elif isinstance(program, Schedule):
        return Schedule(*program.instructions, name=program.name, metadata=program.metadata)
    else:
        raise PulseError(f'Invalid input program {program.__class__.__name__} is specified.')


def inline_subroutines(program: Schedule) -> Schedule:
    """Recursively remove call instructions and inline the respective subroutine instructions.

    Assigned parameter values, which are stored in the parameter table, are also applied.
    The subroutine is copied before the parameter assignment to avoid mutation problem.

    Args:
        program: A program which may contain the subroutine, i.e. ``Call`` instruction.

    Returns:
        A schedule without subroutine.
    """
    schedule = Schedule(name=program.name, metadata=program.metadata)
    for t0, inst in program.instructions:
        if isinstance(inst, instructions.Call):
            # bind parameter
            if bool(inst.arguments):
                subroutine = deepcopy(inst.subroutine)
                subroutine.assign_parameters(value_dict=inst.arguments)
            else:
                subroutine = inst.subroutine
            # recursively inline the program
            inline_schedule = inline_subroutines(subroutine)
            schedule.insert(t0, inline_schedule, inplace=True)
        else:
            schedule.insert(t0, inst, inplace=True)
    return schedule


def remove_directives(schedule: Schedule) -> Schedule:
    """Remove directives.

    Args:
        schedule: A schedule to remove compiler directives.

    Returns:
        A schedule without directives.
    """
    return schedule.exclude(instruction_types=[directives.Directive])


def remove_trivial_barriers(schedule: Schedule) -> Schedule:
    """Remove trivial barriers with 0 or 1 channels.

    Args:
        schedule: A schedule to remove trivial barriers.

    Returns:
        schedule: A schedule without trivial barriers
    """
    def filter_func(inst):
        return (isinstance(inst[1], directives.RelativeBarrier) and
                len(inst[1].channels) < 2)

    return schedule.exclude(filter_func)
