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

"""Replace a schedule with frames by one with instructions on PulseChannels only."""

from typing import Union
import numpy as np

from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.pulse.transforms.resolved_frame import ResolvedFrame, ChannelTracker
from qiskit.pulse.transforms.canonicalization import block_to_schedule
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library.signal import Signal
from qiskit.pulse import channels as chans, instructions
from qiskit.pulse.frame import FramesConfiguration
from qiskit.pulse.instructions import ShiftPhase, ShiftFrequency, Play, SetFrequency, SetPhase


def requires_frame_mapping(schedule: Union[Schedule, ScheduleBlock]) -> bool:
    """Returns True if there are frame instructions or :class:`Signal`s that need mapping.

    Returns:
        True if:
            - There are Signals in the Schedule
            - A SetFrequency, SetPhase, ShiftFrequency, ShiftPhase has a frame that does not
                correspond to any PulseChannel.
    """
    if isinstance(schedule, ScheduleBlock):
        schedule = block_to_schedule(schedule)

    for time, inst in schedule.instructions:
        if isinstance(inst, Play):
            if isinstance(inst.operands[0], Signal):
                return True

        if isinstance(inst, (SetFrequency, SetPhase, ShiftFrequency, ShiftPhase)):
            if inst.channel is None:
                return True

    return False


def map_frames(
    schedule: Union[Schedule, ScheduleBlock], frames_config: FramesConfiguration
) -> Schedule:
    """
    Parse the schedule and replace instructions on Frames that do not have a pulse channel
    by instructions on the appropriate channels. Also replace Signals with play Pulse
    instructions with the appropriate phase/frequency shifts and sets.

    Args:
        schedule: The schedule for which to replace frames with the appropriate
            channels.
        frames_config: An instance of FramesConfiguration defining the frames.

    Returns:
        new_schedule: A new schedule where frames have been replaced with
            their corresponding Drive, Control, and/or Measure channels.

    Raises:
        PulseError: if a frame is not configured.
    """
    if frames_config is None or not frames_config:
        return schedule

    if isinstance(schedule, ScheduleBlock):
        schedule = block_to_schedule(schedule)

    if not requires_frame_mapping(schedule):
        return schedule

    frame_trackers = {}
    for frame, frame_def in frames_config.items():
        frame_trackers[frame] = ResolvedFrame(frame, frame_def, frames_config.sample_duration)

        # Extract shift and set frame operations from the schedule.
        frame_trackers[frame].set_frame_instructions(schedule)

    # Used to keep track of the frequency and phase of the channels
    channel_trackers = {}
    for ch in schedule.channels:
        if isinstance(ch, chans.PulseChannel):
            channel_trackers[ch] = ChannelTracker(
                ch, frames_config[ch.frame].frequency, frames_config.sample_duration
            )

    sched = Schedule(name=schedule.name, metadata=schedule.metadata)

    for time, inst in schedule.instructions:
        if isinstance(inst, Play):

            if inst.frame not in frame_trackers:
                raise PulseError(f"{inst.frame} is not configured and cannot be resolved.")

            # The channel on which the pulse or signal is played.
            chan = inst.channel

            # The frame of the instruction: for a pulse this is simply the channel frame.
            inst_frame = inst.frame

            # The frame to which the instruction will be mapped.
            chan_frame = channel_trackers[chan].frame

            # Get trackers
            frame_tracker = frame_trackers[inst_frame]
            chan_tracker = channel_trackers[chan]

            # Get the current frequency and phase of the frame.
            frame_freq = frame_tracker.frequency(time)
            frame_phase = frame_tracker.phase(time)

            # Get the current frequency and phase of the channel.
            chan_freq = chan_tracker.frequency(time)
            chan_phase = chan_tracker.phase(time)

            # Compute the differences
            freq_diff = frame_freq - chan_freq
            phase_diff = (frame_phase - chan_phase + np.pi) % (2 * np.pi) - np.pi

            if abs(freq_diff) > frame_tracker.tolerance:
                sched.insert(time, ShiftFrequency(freq_diff, chan_frame), inplace=True)

            if abs(phase_diff) > frame_tracker.tolerance:
                sched.insert(time, ShiftPhase(phase_diff, chan_frame), inplace=True)

            # Update the frequency and phase of this channel.
            channel_trackers[chan].set_frequency_phase(time, frame_freq, frame_phase)

            play = Play(inst.pulse, chan)
            sched.insert(time, play, inplace=True)

        # Insert phase and frequency commands that are tied to physical channels.
        elif isinstance(inst, (SetFrequency, ShiftFrequency, SetPhase, ShiftPhase)):
            if inst.channel is not None:
                sched.insert(time, inst, inplace=True)

        elif isinstance(
            inst,
            (
                instructions.Delay,
                instructions.Snapshot,
                instructions.Acquire,
                instructions.Directive,
            ),
        ):
            sched.insert(time, inst, inplace=True)
        elif isinstance(inst, instructions.Call):
            raise PulseError("Inline Call instructions before resolving frames.")
        else:
            raise PulseError(f"Unsupported {inst.__class__.__name__} in frame resolution.")

    return sched
