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

"""Basic transformation functions which take a Schedule (and possibly some arguments) and return
a new Schedule.
"""
import warnings

from typing import List, Optional, Iterable

import numpy as np

from qiskit import pulse
from qiskit.pulse import (Acquire, AcquireInstruction, Delay, Play,
                          InstructionScheduleMap, ScheduleComponent, Schedule)
from qiskit.pulse.passmanager import PassManager
from qiskit.pulse.basepasses import (AnalysisPass, TransformationPass)
from .channels import Channel, AcquireChannel, MeasureChannel, MemorySlot
from .exceptions import PulseError, CompilerError


class AlignMeasuresAnalysis(AnalysisPass):
    def __init__(
        self,
        inst_map: InstructionScheduleMap,
        cal_gate: str = 'u3',
        max_calibration_duration: int = None,
    ):
        """Calculate the latest time to align all measures in a program."""
        super().__init__()
        self.inst_map = inst_map
        self.cal_gate = cal_gate
        self.max_calibration_duration = max_calibration_duration

    def run(
        self,
        program: pulse.Program,
    ) -> pulse.Program:
        """Return the the max between the duration of the calibration time and
        the absolute time of the latest scheduled acquire."""
        if self.max_calibration_duration is None:
            align_time = self.get_max_calibration_duration()
        else:
            align_time = self.max_calibration_duration
        for schedule in program.schedules:
            last_acquire = 0
            acquire_times = [time for time, inst in schedule.instructions
                             if isinstance(inst, (Acquire, AcquireInstruction))]
            if acquire_times:
                last_acquire = max(acquire_times)
            align_time = max(align_time, last_acquire)
        self.analysis.meas_align_time = align_time
        return program

    def get_max_calibration_duration(self):
        """Return the time needed to allow for readout discrimination calibration pulses."""
        max_calibration_duration = 0
        for qubits in self.inst_map.qubits_with_instruction(self.cal_gate):
            cmd = self.inst_map.get(self.cal_gate, qubits, np.pi, 0, np.pi)
            max_calibration_duration = max(cmd.duration, max_calibration_duration)
        return max_calibration_duration


class AlignMeasures(TransformationPass):
    """Return an program with all measurements occurring at the same physical time.

    Minimum measurement wait time (to allow for calibration pulses) is enforced.
    """

    def __init__(
        self,
        inst_map: Optional[InstructionScheduleMap] = None,
        cal_gate: str = 'u3',
        max_calibration_duration: Optional[int] = None,
        align_time: Optional[int] = None,
    ):
        """Return new schedules where measurements occur at the same physical time.

        Args:
            inst_map: Mapping of circuit operations to pulse schedules
            cal_gate: The name of the gate to inspect for the calibration time
            max_calibration_duration: If provided, inst_map and cal_gate will be ignored
            align_time: If provided, this will be used as final align time.
        """
        super().__init__()
        if align_time is None and max_calibration_duration is None and inst_map is None:
            raise CompilerError("Must provide a inst_map, an alignment time, or a calibration duration.")
        if align_time is not None and align_time < 0:
            raise CompilerError("Align time cannot be negative.")
        if align_time is None:
            self.requires.append(AlignMeasuresAnalysis(
                inst_map, cal_gate, max_calibration_duration=max_calibration_duration))

        self.align_time = align_time

    def run(self, program: pulse.Program):
        align_time = self.align_time or self.analysis.meas_align_time
        # Shift acquires according to the new scheduled time
        for idx, schedule in enumerate(program.schedules):
            new_schedule = Schedule(name=schedule.name)
            acquired_channels = set()
            measured_channels = set()

            for time, inst in schedule.instructions:
                for chan in inst.channels:
                    if isinstance(chan, MeasureChannel):
                        if chan.index in measured_channels:
                            raise PulseError(
                                "Multiple measurements are not supported by this "
                                "rescheduling pass.")
                    elif chan.index in acquired_channels:
                        raise PulseError(
                            "Pulse encountered on channel {0} after acquire on "
                            "same channel.".format(chan.index))

                if isinstance(inst, (Acquire, AcquireInstruction)):
                    if time > align_time:
                        warnings.warn(
                            "You provided an align_time which is scheduling an acquire "
                            "sooner than it was scheduled for in the original Schedule.")
                    new_schedule |= inst << align_time
                    acquired_channels.add(inst.channel.index)
                elif isinstance(inst.channels[0], MeasureChannel):
                    new_schedule |= inst << align_time
                    measured_channels.update({a.index for a in inst.channels})
                else:
                    new_schedule |= inst << time

            program.replace_schedule(idx, new_schedule)

        return program


def align_measures(schedules: Iterable[ScheduleComponent],
                   inst_map: Optional[InstructionScheduleMap] = None,
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
        cal_gate: The name of the gate to inspect for the calibration time
        max_calibration_duration: If provided, inst_map and cal_gate will be ignored
        align_time: If provided, this will be used as final align time.

    Returns:
        Schedule

    Raises:
        PulseError: if an acquire or pulse is encountered on a channel that has already been part
                    of an acquire, or if align_time is negative
    """
    pm = PassManager()
    pm.append(AlignMeasures(
        inst_map,
        cal_gate,
        max_calibration_duration,
        align_time,
        ),
    )

    return pm.run(pulse.Program(schedules=schedules)).schedules


class AddImplicitAcquires(TransformationPass):
        def __init__(
            self,
            meas_map: List[List[int]]
        ):
            """Transformation pass that returns a new pulse program with missing acquires
            added as defined by the ``meas_map``.

            .. warning:: Since new acquires are being added, Memory Slots will be set to match the
                         qubit index. This may overwrite your specification.

            Args:
                meas_map: List of lists of qubits that are measured together.

            Returns:
                A ``Schedule`` with the additional acquisition commands.
            """
            super().__init__()
            self.meas_map = meas_map

        def run(self, program: pulse.Program):
            for idx, schedule in enumerate(program.schedules):
                new_schedule = Schedule(name=schedule.name)
                acquire_map = dict()

                for time, inst in schedule.instructions:
                    if isinstance(inst, (Acquire, AcquireInstruction)):
                        if inst.mem_slot and inst.mem_slot.index != inst.channel.index:
                            warnings.warn("One of your acquires was mapped to a memory slot which didn't match"
                                          " the qubit index. I'm relabeling them to match.")

                        # Get the label of all qubits that are measured with the qubit(s) in this instruction
                        all_qubits = []
                        for sublist in self.meas_map:
                            if inst.channel.index in sublist:
                                all_qubits.extend(sublist)
                        # Replace the old acquire instruction by a new one explicitly acquiring all qubits in
                        # the measurement group.
                        for i in all_qubits:
                            explicit_inst = Acquire(inst.duration, AcquireChannel(i),
                                                    mem_slot=MemorySlot(i),
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

                program.replace_schedule(idx, new_schedule)
            return program


def add_implicit_acquires(
    schedule: ScheduleComponent,
    meas_map: List[List[int]],
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
    pm = PassManager()
    pm.append(AddImplicitAcquires(
        meas_map,
        ),
    )

    return pm.run(pulse.Program(schedules=schedule)).schedules[0]


class PadProgram(TransformationPass):
        def __init__(
            self,
            channels: Optional[Iterable[Channel]] = None,
            until: Optional[int] = None,
        ):
            """Transformation pass that pads empty times in a program with delays.

            Args:
                meas_map: List of lists of qubits that are measured together.

            Returns:
                A ``Schedule`` with the additional acquisition commands.
            """
            super().__init__()
            self.channels = channels
            self.until = until

        def run(self, program: pulse.Program):
            for idx, schedule in enumerate(program.schedules):
                until = self.until or schedule.duration
                channels = self.channels or schedule.channels

                until = until or schedule.duration
                channels = channels or schedule.channels

                for channel in channels:
                    if channel not in schedule.channels:
                        schedule |= Delay(until, channel)
                        continue

                    curr_time = 0
                    # TODO: Replace with method of getting instructions on a channel
                    for interval in schedule.timeslots[channel]:
                        if curr_time >= until:
                            break
                        if interval[0] != curr_time:
                            end_time = min(interval[0], until)
                            schedule = schedule.insert(curr_time, Delay(end_time - curr_time, channel))
                        curr_time = interval[1]
                    if curr_time < until:
                        schedule = schedule.insert(curr_time, Delay(until - curr_time, channel))

                program.replace_schedule(idx, schedule)
            return program


def pad(
    schedule: Schedule,
    channels: Optional[Iterable[Channel]] = None,
    until: Optional[int] = None,
) -> Schedule:
    """Pad the input ``Schedule`` with ``Delay`` s on all unoccupied timeslots until ``until``
    if it is provided, otherwise until ``schedule.duration``.

    Args:
        schedule: Schedule to pad.
        channels: Channels to pad. Defaults to all channels in ``schedule`` if not provided.
                  If the supplied channel is not a member of ``schedule``, it will be added.
        until: Time to pad until. Defaults to ``schedule.duration`` if not provided.

    Returns:
        The padded schedule.
    """
    pm = PassManager()
    pm.append(PadProgram(
        channels,
        until
        ),
    )
    return pm.run(pulse.Program(schedules=schedule)).schedules[0]


class CompressPulses(TransformationPass):
    """Transformation pass to replace identical pulses."""

    def run(self, program: pulse.Program):
        existing_pulses = []
        for sched_idx, schedule in enumerate(program.schedules):
            new_schedule = Schedule(name=schedule.name)

            for time, inst in schedule.instructions:
                if isinstance(inst, Play):
                    if inst.pulse in existing_pulses:
                        idx = existing_pulses.index(inst.pulse)
                        identical_pulse = existing_pulses[idx]
                        new_schedule |= Play(identical_pulse, inst.channel, inst.name) << time
                    else:
                        existing_pulses.append(inst.pulse)
                        new_schedule |= inst << time
                else:
                    new_schedule |= inst << time

            program.replace_schedule(sched_idx, new_schedule)
        return program


def compress_pulses(schedules: List[Schedule]) -> List[Schedule]:
    """Optimization pass to replace identical pulses.

    Args:
        schedules: Schedules to compress.

    Returns:
        Compressed schedules.
    """
    pm = PassManager()
    pm.append(CompressPulses())
    return pm.run(pulse.Program(schedules=schedules)).schedules
