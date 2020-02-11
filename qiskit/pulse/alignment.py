# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Alignment methods."""
from typing import List, Union

import numpy as np

from qiskit import pulse
from qiskit.pulse.reschedule import pad


def push_append(this: List[pulse.ScheduleComponent],
                other: List[pulse.ScheduleComponent]) -> pulse.Schedule:
        r"""Return a new schedule with `schedule` inserted at the maximum time over
        all channels shared between `self` and `schedule`.

       $t = \textrm{max}({x.stop\_time |x \in self.channels \cap schedule.channels})$

        Args:
            schedule: schedule to be appended
            buffer: Whether to obey buffer when appending
        """
        channels = list(set(this.channels) & set(other.channels))

        ch_slacks = [this.stop_time - this.ch_stop_time(channel) + other.ch_start_time(channel)
                     for channel in channels]

        if ch_slacks:
            slack_chan = channels[np.argmin(ch_slacks)]
            insert_time = this.ch_stop_time(slack_chan) - other.ch_start_time(slack_chan)
        else:
            insert_time = 0
        return this.insert(insert_time, other)


def left_align(*instructions: List[Union[pulse.Instruction, pulse.Schedule]]) -> pulse.Schedule:
    """Align a list of pulse instructions on the left.

    Args:
        instructions: List of pulse instructions to align.

    Returns:
        pulse.Schedule
    """
    aligned = pulse.Schedule()
    for instruction in instructions:
        aligned = push_append(aligned, instruction)

    return aligned


def left_barrier(*instructions: List[pulse.ScheduleComponent], channels=None) -> pulse.Schedule:
    """Align on the left and create a barrier so that pulses cannot be inserted
        within this pulse interval.

    Args:
        instructions: List of pulse instructions to align.

    Returns:
        pulse.Schedule
    """
    aligned = left_align(*instructions)
    return pad(aligned, channels=channels)


def right_align(*instructions: List[pulse.ScheduleComponent]) -> pulse.Schedule:
    """Align a list of pulse instructions on the right.

    Args:
        instructions: List of pulse instructions to align.

    Returns:
        pulse.Schedule
    """
    left_aligned = left_align(*instructions)
    max_duration = 0

    channel_durations = {}
    for channel in left_aligned.channels:
        channel_sched = left_aligned.filter(channels=[channel])
        channel_duration = channel_sched.duration-channel_sched.start_time
        channel_durations[channel] = channel_sched.duration
        max_duration = max(max_duration, channel_duration)

    aligned = pulse.Schedule()
    for instr_time, instruction in left_aligned.instructions:
        instr_max_dur = max(channel_durations[channel] for channel in instruction.channels)
        instr_delayed_time = max_duration - instr_max_dur + instr_time
        aligned.insert(instr_delayed_time, instruction, mutate=True)

    return aligned


def right_barrier(*instructions: List[pulse.ScheduleComponent], channels=None) -> pulse.Schedule:
    """Align on the right and create a barrier so that pulses cannot be
        inserted within this pulse interval.

    Args:
        instructions: List of pulse instructions to align.

    Returns:
        pulse.Schedule
    """
    aligned = right_align(*instructions)
    return pad(aligned, channels=channels)


def align_in_sequence(*instructions: List[pulse.ScheduleComponent]) -> pulse.Schedule:
    """Align a list of pulse instructions sequentially in time.
    Args:
        instructions: List of pulse instructions to align.
    Returns:
        A new pulse schedule with instructions`
    """
    aligned = pulse.Schedule()
    for instruction in instructions:
        aligned.insert(aligned.duration, instruction, mutate=True)
    return aligned
