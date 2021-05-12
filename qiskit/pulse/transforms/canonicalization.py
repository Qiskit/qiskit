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

import copy
import warnings
from collections import defaultdict
from typing import List, Optional, Iterable, Union

from qiskit.pulse import channels as chans, exceptions, instructions
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.exceptions import UnassignedDurationError
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.instructions import directives
from qiskit.pulse.library import Waveform
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.pulse.transforms.alignments import Frozen


def block_to_schedule(block: ScheduleBlock) -> Schedule:
    """Convert ``ScheduleBlock`` into ``Schedule``.

    Args:
        block: A ``ScheduleBlock`` to convert.

    Returns:
        A program in ``Schedule`` representation.

    .. note:: This transformation only converts blocks in the main program.
        Blocks inline in ``Call`` instructions are not converted.
        Apply `qiskit.pulse.transforms.inline_subroutines` transformation first.
    """
    schedule = Schedule.initialize_from(block)
    if not block.scheduled():
        block = allocate_node_time(block)

    for node in block.nodes():
        op_data = node.data
        if isinstance(op_data, ScheduleBlock):
            op_data = block_to_schedule(op_data)
        schedule.add_node(op_data, node.time)

    return schedule


def schedule_to_block(schedule: Schedule) -> ScheduleBlock:
    """Convert ``Schedule`` into ``ScheduleBlock``.

    Args:
        schedule: A ``Schedule`` to convert.

    Returns:
        A program in ``ScheduleBlock`` representation.

    .. note:: This transformation only converts schedules in the main program.
        Schedules inline in ``Call`` instructions are not converted.
        Apply `qiskit.pulse.transforms.inline_subroutines` transformation first.
    """
    block = ScheduleBlock.initialize_from(schedule)
    block._alignment_context = Frozen()

    for node in block.nodes():
        op_data = node.data
        if isinstance(op_data, Schedule):
            op_data = schedule_to_block(op_data)
        block.add_node(op_data, node.time)

    return block


def allocate_node_time(block: ScheduleBlock) -> ScheduleBlock:
    """Schedule block nodes.

    .. notes::
        This is mutably update node time of input block.
        Previously assigned node time is removed before allocating new node time.

    Args:
        block: A block to schedule.

    Returns:
        A scheduled block.

    Raises:
        UnassignedDurationError: When any instruction duration is not assigned.
    """
    if not block.is_schedulable():
        raise UnassignedDurationError(
            "All instruction durations should be assigned before creating `Schedule`."
            "Please check `.parameters` to find unassigned parameter objects."
        )

    if block.scheduled() or isinstance(block.alignment_context, Frozen):
        # already scheduled
        return block

    # schedule nested blocks to extract the node duration
    for node in block.nodes(flatten=True):
        if isinstance(node.data, ScheduleBlock):
            allocate_node_time(node.data)

    return block.alignment_context.align(block, inplace=True)


def compress_pulses(
    schedules: List[Union[Schedule, ScheduleBlock]]
) -> List[Union[Schedule, ScheduleBlock]]:
    """Optimization pass to replace identical waveforms.

    Args:
        schedules: Schedules to compress.

    Returns:
        Compressed schedules.

    .. note::
        This transform doesn't check nested schedule and subroutines.
        These transformations should be applied first.
    """
    existing_pulses = []
    new_schedules = []

    for schedule in schedules:
        new_schedule = schedule.__class__.initialize_from(schedule)

        for node in schedule.nodes():
            op_data = node.data
            if isinstance(op_data, instructions.Play):
                if isinstance(op_data.pulse, Waveform):
                    # Waveform should be merged to remove duplicated pulse library entries
                    if op_data.pulse in existing_pulses:
                        idx = existing_pulses.index(op_data.pulse)
                        new_op_data = instructions.Play(
                            pulse=existing_pulses[idx], channel=op_data.channel, name=op_data.name
                        )
                    else:
                        existing_pulses.append(op_data.pulse)
                        new_op_data = op_data
                else:
                    # Parametric pulse doesn't consume memory
                    new_op_data = op_data
                new_schedule.add_node(new_op_data, node.time)
            else:
                new_schedule.add_node(op_data, node.time)

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

    .. notes::
        ``ScheduleBlock`` cannot be flattened because a block cannot be inserted into
        another block with different alignment context.
        This causes a fatal problem for scheduling.
    """
    if isinstance(program, Schedule):
        flat_sched = Schedule.initialize_from(program)
        for time, inst in program.instructions:
            flat_sched.insert(time, inst, inplace=True)
        return flat_sched
    else:
        raise PulseError(f"Invalid input program {program.__class__.__name__} is specified.")


def inline_subroutines(
    program: Union[Schedule, ScheduleBlock], inplace: bool = False
) -> Union[Schedule, ScheduleBlock]:
    """Recursively remove call instructions and inline the respective subroutine instructions.

    Assigned parameter values, which are stored in the parameter table, are also applied.
    The subroutine is copied before the parameter assignment to avoid mutation problem.

    Args:
        program: A program which may contain the subroutine, i.e. ``Call`` instruction.
        inplace: Set ``True`` to override input schedule.

    Returns:
        A schedule without subroutine.

    Raises:
        PulseError: When subroutine is not valid data format.
    """
    if not inplace:
        main_routine = copy.deepcopy(program)
    else:
        main_routine = program

    for node in main_routine.nodes():
        op_data = node.data
        if isinstance(op_data, instructions.Call):
            subroutine = op_data.assigned_subroutine()
            if not isinstance(subroutine, type(main_routine)):
                # Format data if data type is not identical to main program.
                if isinstance(main_routine, Schedule):
                    subroutine = block_to_schedule(subroutine)
                elif isinstance(main_routine, ScheduleBlock):
                    subroutine = schedule_to_block(subroutine)
                else:
                    raise PulseError(
                        f"Invalid subroutine data type {subroutine.__class__.__name__} is found."
                    )
            # recursively inline the program
            node.data = inline_subroutines(subroutine, inplace=True)
        if isinstance(op_data, (Schedule, ScheduleBlock)):
            # recursively inline the program
            node.data = inline_subroutines(op_data, inplace=True)

    return main_routine


def remove_directives(
    schedule: Union[Schedule, ScheduleBlock],
) -> Union[Schedule, ScheduleBlock]:
    """Remove directives.

    Args:
        schedule: A schedule to remove compiler directives.

    Returns:
        A schedule without directives.
    """
    from qiskit.pulse.filters import filter_instructions, with_instruction_types

    return filter_instructions(
        schedule, with_instruction_types([directives.Directive]), negate=True
    )


def remove_trivial_barriers(
    schedule: Union[Schedule, ScheduleBlock],
) -> Union[Schedule, ScheduleBlock]:
    """Remove trivial barriers with 0 or 1 channels.

    Args:
        schedule: A schedule to remove trivial barriers.

    Returns:
        schedule: A schedule without trivial barriers
    """
    from qiskit.pulse.filters import filter_instructions

    return filter_instructions(
        schedule,
        lambda node: isinstance(node[1], directives.RelativeBarrier) and len(node[1].channels < 2),
        negate=True,
    )


def align_measures(
    schedules: Iterable[Union[Schedule, ScheduleBlock]],
    inst_map: Optional[InstructionScheduleMap] = None,
    cal_gate: str = "u3",
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

        d0 = pulse.DriveChannel(0)
        m0 = pulse.MeasureChannel(0)
        a0 = pulse.AcquireChannel(0)
        mem0 = pulse.MemorySlot(0)

        sched = pulse.Schedule()
        sched.append(pulse.Play(pulse.Constant(10, 0.5), d0), inplace=True)
        sched.append(pulse.Play(pulse.Constant(10, 1.), m0).shift(sched.duration), inplace=True)
        sched.append(pulse.Acquire(20, a0, mem0).shift(sched.duration), inplace=True)

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
        PulseError:
            - If the provided alignment time is negative.
    """
    _aligned_channels = (chans.MeasureChannel, chans.AcquireChannel)

    if align_time is not None and align_time < 0:
        raise exceptions.PulseError("Align time cannot be negative.")

    # get first acquire times
    first_acquire_times = []
    for schedule in schedules:
        if isinstance(schedule, ScheduleBlock) and not schedule.scheduled():
            schedule = allocate_node_time(schedule)

        qubit_first_acquire_times = defaultdict(lambda: None)
        for time, op_data in inline_subroutines(schedule, inplace=False).instructions:
            if (
                isinstance(op_data, instructions.Acquire)
                and op_data.channel.index not in qubit_first_acquire_times
            ):
                qubit_first_acquire_times[op_data.channel.index] = time
        first_acquire_times.append(qubit_first_acquire_times)

    # maximum calibration duration
    if max_calibration_duration is None:
        max_calibration_duration = 0
        if inst_map is not None:
            for qubits in inst_map.qubits_with_instruction(cal_gate):
                cmd = inst_map.get(cal_gate, qubits)
                max_calibration_duration = max(cmd.duration, max_calibration_duration)

    # Extract the maximum acquire in every schedule across all acquires in the schedule.
    # If there are no acquires in the schedule default to 0.
    max_acquire_times = [max(0, *times.values()) for times in first_acquire_times]

    if align_time is None:
        align_time = max(max_calibration_duration, *max_acquire_times)

    # Shift acquires according to the new scheduled time
    new_schedules = []
    for sched_idx, schedule in enumerate(schedules):
        new_schedule = schedule.__class__.initialize_from(schedule)
        stop_time = schedule.stop_time

        # shift for instructions before measurement
        if align_all:
            if first_acquire_times[sched_idx]:
                shift = align_time - max_acquire_times[sched_idx]
            else:
                shift = align_time - stop_time
        else:
            shift = 0

        for time, op_data in schedule.instructions:
            if all(isinstance(chan, _aligned_channels) for chan in op_data.channels):
                sched_first_acquire_times = first_acquire_times[sched_idx]
                max_start_time = max(
                    sched_first_acquire_times.get(chan.index, 0) for chan in op_data.channels
                )
                shift = align_time - max_start_time

            if shift < 0:
                warnings.warn(
                    "The provided alignment time is scheduling an acquire instruction "
                    "earlier than it was scheduled for in the original Schedule. "
                    "This may result in an instruction being scheduled before t=0 and "
                    "an error being raised."
                )
            new_schedule.add_node(op_data, time + shift)
        new_schedules.append(new_schedule)

    return new_schedules


def add_implicit_acquires(
    schedule: Union[Schedule, ScheduleBlock], meas_map: List[List[int]]
) -> Union[Schedule, ScheduleBlock]:
    """Return a new schedule with implicit acquires from the measurement mapping replaced by
    explicit ones.

    .. warning:: Since new acquires are being added, Memory Slots will be set to match the
                 qubit index. This may overwrite your specification.

    Args:
        schedule: Schedule to be aligned.
        meas_map: List of lists of qubits that are measured together.
        inplace: Set ``True`` to override input schedule.

    Returns:
        A ``Schedule`` with the additional acquisition instructions.
    """
    new_schedule = schedule.__class__.initialize_from(schedule)

    for node in schedule.nodes():
        op_data = node.data
        if isinstance(op_data, instructions.Acquire):
            warnings.warn(
                f"Acquire instruction at {node.location} was mapped to a memory slot "
                "which didn't match the qubit index. I'm relabeling them to match."
            )
            # Get the label of all qubits that are measured with the qubit(s) in this instruction
            all_qubits = []
            for sublist in meas_map:
                if op_data.channel.index in sublist:
                    all_qubits.extend(sublist)
            # Replace the old acquire instruction by a new one explicitly acquiring all qubits in
            # the measurement group.
            for i in all_qubits:
                explicit_op_data = instructions.Acquire(
                    duration=op_data.duration,
                    channel=chans.AcquireChannel(i),
                    mem_slot=chans.MemorySlot(i),
                    kernel=op_data.kernel,
                    discriminator=op_data.discriminator,
                )
                new_schedule.add_node(explicit_op_data, node.time)
        elif isinstance(op_data, (Schedule, ScheduleBlock)):
            # recursively add implicit acquires to nested schedules
            new_schedule.add_node(add_implicit_acquires(op_data, meas_map), node.time)
        else:
            new_schedule.add_node(op_data, node.time)

    return new_schedule


def pad(
    schedule: Union[Schedule, ScheduleBlock],
    channels: Optional[Iterable[chans.Channel]] = None,
    until: Optional[int] = None,
    inplace: bool = False,
) -> Schedule:
    """Pad the input Schedule with ``Delay``s on all unoccupied timeslots until
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
    if not inplace:
        schedule = copy.deepcopy(schedule)

    if isinstance(schedule, ScheduleBlock) and not schedule.scheduled():
        schedule = allocate_node_time(schedule)

    until = until or schedule.duration
    channels = channels or schedule.channels

    for channel in channels:
        if channel not in schedule.channels:
            schedule.add_node(instructions.Delay(until, channel), time=0)
            continue

        curr_time = 0
        for time, op_data in schedule.instructions:
            if curr_time >= until:
                break
            if time != curr_time:
                schedule.add_node(
                    instructions.Delay(min(time, until) - curr_time, channel), time=curr_time
                )
                curr_time = time + op_data.duration
        if curr_time < until:
            schedule.add_node(instructions.Delay(until - curr_time, channel), time=curr_time)

    return schedule
