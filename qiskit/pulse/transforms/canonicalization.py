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

import warnings
from collections import defaultdict
from typing import List, Optional, Iterable, Union

import numpy as np

from qiskit.pulse import channels as chans, exceptions, instructions
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.exceptions import UnassignedDurationError
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.instructions import directives
from qiskit.pulse.schedule import Schedule, ScheduleBlock, ScheduleComponent


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
    return block.alignment_context.align(schedule)


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


def flatten(program: Schedule) -> Schedule:
    """Flatten (inline) any called nodes into a Schedule tree with no nested children.

    Args:
        program: Pulse program to remove nested structure.

    Returns:
        Flatten pulse program.

    Raises:
        PulseError: When invalid data format is given.
    """
    if isinstance(program, Schedule):
        return Schedule(*program.instructions, name=program.name, metadata=program.metadata)
    else:
        raise PulseError(f'Invalid input program {program.__class__.__name__} is specified.')


def inline_subroutines(program: Union[Schedule, ScheduleBlock]) -> Union[Schedule, ScheduleBlock]:
    """Recursively remove call instructions and inline the respective subroutine instructions.

    Assigned parameter values, which are stored in the parameter table, are also applied.
    The subroutine is copied before the parameter assignment to avoid mutation problem.

    Args:
        program: A program which may contain the subroutine, i.e. ``Call`` instruction.

    Returns:
        A schedule without subroutine.

    Raises:
        PulseError: When input program is not valid data format.
    """
    if isinstance(program, Schedule):
        return _inline_schedule(program)
    elif isinstance(program, ScheduleBlock):
        return _inline_block(program)
    else:
        raise PulseError(f'Invalid program {program.__class__.__name__} is specified.')


def _inline_schedule(schedule: Schedule) -> Schedule:
    """A helper function to inline subroutine of schedule.

    .. note:: If subroutine is ``ScheduleBlock`` it is converted into Schedule to get ``t0``.
    """
    ret_schedule = Schedule(name=schedule.name,
                            metadata=schedule.metadata)
    for t0, inst in schedule.instructions:
        if isinstance(inst, instructions.Call):
            # bind parameter
            subroutine = inst.assigned_subroutine()
            # convert into schedule if block is given
            if isinstance(subroutine, ScheduleBlock):
                subroutine = block_to_schedule(subroutine)
            # recursively inline the program
            inline_schedule = _inline_schedule(subroutine)
            ret_schedule.insert(t0, inline_schedule, inplace=True)
        else:
            ret_schedule.insert(t0, inst, inplace=True)
    return ret_schedule


def _inline_block(block: ScheduleBlock) -> ScheduleBlock:
    """A helper function to inline subroutine of schedule block.

    .. note:: If subroutine is ``Schedule`` the function raises an error.
    """
    ret_block = ScheduleBlock(alignment_context=block.alignment_context,
                              name=block.name,
                              metadata=block.metadata)
    for inst in block.instructions:
        if isinstance(inst, instructions.Call):
            # bind parameter
            subroutine = inst.assigned_subroutine()
            if isinstance(subroutine, Schedule):
                raise PulseError(f'A subroutine {subroutine.name} is a pulse Schedule. '
                                 'This program cannot be inserted into ScheduleBlock because '
                                 't0 associated with instruction will be lost.')
            # recursively inline the program
            inline_block = _inline_block(subroutine)
            ret_block.append(inline_block, inplace=True)
        else:
            ret_block.append(inst, inplace=True)
    return ret_block


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


def align_measures(schedules: Iterable[ScheduleComponent],
                   inst_map: Optional[InstructionScheduleMap] = None,
                   cal_gate: str = 'u3',
                   max_calibration_duration: Optional[int] = None,
                   align_time: Optional[int] = None,
                   align_all: Optional[bool] = True,
                   ) -> List[Schedule]:
    """Return new schedules where measurements occur at the same physical time.

    This transformation will align the first :class:`qiskit.pulse.Acquire` on
    every channel to occur at the same time.

    Minimum measurement wait time (to allow for calibration pulses) is enforced
    and may be set with ``max_calibration_duration``.

    By default only instructions containing a :class:`~qiskit.pulse.AcquireChannel`
    or :class:`~qiskit.pulse.MeasureChannel` will be shifted. If you wish to keep
    the relative timing of all instructions in the schedule set ``align_all=True``.

    This method assumes that ``MeasureChannel(i)`` and ``AcquireChannel(i)``
    correspond to the same qubit and the acquire/play instructions
    should be shifted together on these channels.

    .. jupyter-kernel:: python3
        :id: align_measures

    .. jupyter-execute::

        from qiskit import pulse
        from qiskit.pulse import transforms

        with pulse.build() as sched:
            with pulse.align_sequential():
                pulse.play(pulse.Constant(10, 0.5), pulse.DriveChannel(0))
                pulse.play(pulse.Constant(10, 1.), pulse.MeasureChannel(0))
                pulse.acquire(20, pulse.AcquireChannel(0), pulse.MemorySlot(0))

        sched_shifted = sched << 20

        aligned_sched, aligned_sched_shifted = transforms.align_measures([sched, sched_shifted])

        assert aligned_sched == aligned_sched_shifted

    If it is desired to only shift acquisition and measurement stimulus instructions
    set the flag ``align_all=False``:

    .. jupyter-execute::

        aligned_sched, aligned_sched_shifted = transforms.align_measures(
            [sched, sched_shifted],
            align_all=False,
        )

        assert aligned_sched != aligned_sched_shifted


    Args:
        schedules: Collection of schedules to be aligned together
        inst_map: Mapping of circuit operations to pulse schedules
        cal_gate: The name of the gate to inspect for the calibration time
        max_calibration_duration: If provided, inst_map and cal_gate will be ignored
        align_time: If provided, this will be used as final align time.
        align_all: Shift all instructions in the schedule such that they maintain
            their relative alignment with the shifted acquisition instruction.
            If ``False`` only the acquisition and measurement pulse instructions
            will be shifted.
    Returns:
        The input list of schedules transformed to have their measurements aligned.

    Raises:
        PulseError: If the provided alignment time is negative.
    """
    def get_first_acquire_times(schedules):
        """Return a list of first acquire times for each schedule."""
        acquire_times = []
        for schedule in schedules:
            visited_channels = set()
            qubit_first_acquire_times = defaultdict(lambda: None)

            for time, inst in schedule.instructions:
                if (isinstance(inst, instructions.Acquire) and
                        inst.channel not in visited_channels):
                    visited_channels.add(inst.channel)
                    qubit_first_acquire_times[inst.channel.index] = time

            acquire_times.append(qubit_first_acquire_times)
        return acquire_times

    def get_max_calibration_duration(inst_map, cal_gate):
        """Return the time needed to allow for readout discrimination calibration pulses."""
        # TODO (qiskit-terra #5472): fix behavior of this.
        max_calibration_duration = 0
        for qubits in inst_map.qubits_with_instruction(cal_gate):
            cmd = inst_map.get(cal_gate, qubits, np.pi, 0, np.pi)
            max_calibration_duration = max(cmd.duration, max_calibration_duration)
        return max_calibration_duration

    if align_time is not None and align_time < 0:
        raise exceptions.PulseError("Align time cannot be negative.")

    first_acquire_times = get_first_acquire_times(schedules)
    # Extract the maximum acquire in every schedule across all acquires in the schedule.
    # If there are no acquires in the schedule default to 0.
    max_acquire_times = [max(0, *times.values()) for times in first_acquire_times]
    if align_time is None:
        if max_calibration_duration is None:
            if inst_map:
                max_calibration_duration = get_max_calibration_duration(inst_map, cal_gate)
            else:
                max_calibration_duration = 0
        align_time = max(max_calibration_duration, *max_acquire_times)

    # Shift acquires according to the new scheduled time
    new_schedules = []
    for sched_idx, schedule in enumerate(schedules):
        new_schedule = Schedule(name=schedule.name, metadata=schedule.metadata)
        stop_time = schedule.stop_time

        if align_all:
            if first_acquire_times[sched_idx]:
                shift = align_time - max_acquire_times[sched_idx]
            else:
                shift = align_time - stop_time
        else:
            shift = 0

        for time, inst in schedule.instructions:
            measurement_channels = {
                chan.index for chan in inst.channels if
                isinstance(chan, (chans.MeasureChannel, chans.AcquireChannel))
            }
            if measurement_channels:
                sched_first_acquire_times = first_acquire_times[sched_idx]
                max_start_time = max(sched_first_acquire_times[chan]
                                     for chan in measurement_channels if
                                     chan in sched_first_acquire_times)
                shift = align_time - max_start_time

            if shift < 0:
                warnings.warn(
                    "The provided alignment time is scheduling an acquire instruction "
                    "earlier than it was scheduled for in the original Schedule. "
                    "This may result in an instruction being scheduled before t=0 and "
                    "an error being raised."
                )
            new_schedule.insert(time+shift, inst, inplace=True)

        new_schedules.append(new_schedule)

    return new_schedules


def add_implicit_acquires(schedule: ScheduleComponent,
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
    new_schedule = Schedule(name=schedule.name, metadata=schedule.metadata)
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
                                                     discriminator=inst.discriminator)
                if time not in acquire_map:
                    new_schedule.insert(time, explicit_inst, inplace=True)
                    acquire_map = {time: {i}}
                elif i not in acquire_map[time]:
                    new_schedule.insert(time, explicit_inst, inplace=True)
                    acquire_map[time].add(i)
        else:
            new_schedule.insert(time, inst, inplace=True)

    return new_schedule
