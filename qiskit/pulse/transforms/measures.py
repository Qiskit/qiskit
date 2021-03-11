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
"""A collection of functions to reschedule measurement sequences."""

import warnings
from collections import defaultdict
from typing import List, Optional, Iterable

import numpy as np

from qiskit.pulse import channels as chans, exceptions, instructions
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.schedule import Schedule, ScheduleComponent


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
