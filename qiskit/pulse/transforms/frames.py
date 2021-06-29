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

from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.pulse.transforms.resolved_frame import ResolvedFrame, ChannelTracker
from qiskit.pulse.transforms.canonicalization import block_to_schedule
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library.signal import Signal
from qiskit.pulse import channels as chans, instructions
from qiskit.pulse.frame import FramesConfiguration
from qiskit.pulse.instructions import ShiftPhase, ShiftFrequency, Play, SetFrequency, SetPhase


def requires_frame_mapping(schedule: Schedule) -> bool:
    """Returns True if there are frame instructions or :class:`Signal`s that need mapping.

    Returns:
        True if:
            - There are Signals in the Schedule
            - A SetFrequency, SetPhase, ShiftFrequency, ShiftPhase has a frame that does not
                correspond to any PulseChannel.
    """
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

    resolved_frames = {}
    for frame, frame_def in frames_config.items():
        resolved_frames[frame] = ResolvedFrame(frame, frame_def, frames_config.sample_duration)

        # Extract shift and set frame operations from the schedule.
        resolved_frames[frame].set_frame_instructions(schedule)

    # Used to keep track of the frequency and phase of the channels
    channel_trackers = {}
    for ch in schedule.channels:
        if isinstance(ch, chans.PulseChannel):
            channel_trackers[ch] = ChannelTracker(ch, frames_config.sample_duration)

    sched = Schedule(name=schedule.name, metadata=schedule.metadata)

    for time, inst in schedule.instructions:
        if isinstance(inst, Play):
            chan = inst.channel
            chan_frame = channel_trackers[chan].frame

            if inst.frame not in resolved_frames:
                raise PulseError(f"{inst.frame} is not configured and cannot " f"be resolved.")

            resolved_frame = resolved_frames[inst.frame]

            frame_freq = resolved_frame.frequency(time)
            frame_phase = resolved_frame.phase(time)

            # If the frequency and phase of the channel has already been set once in
            # the past we compute shifts.
            if channel_trackers[chan].is_initialized():
                freq_diff = frame_freq - channel_trackers[chan].frequency(time)
                phase_diff = frame_phase - channel_trackers[chan].phase(time)

                if abs(freq_diff) > resolved_frame.tolerance:
                    sched.insert(time, ShiftFrequency(freq_diff, chan_frame), inplace=True)

                if abs(phase_diff) > resolved_frame.tolerance:
                    sched.insert(time, ShiftPhase(phase_diff, chan_frame), inplace=True)

            # If the channel's phase and frequency has not been set in the past
            # we set it now
            else:
                if frame_freq != 0:
                    sched.insert(time, SetFrequency(frame_freq, chan_frame), inplace=True)

                if frame_phase != 0:
                    sched.insert(time, SetPhase(frame_phase, chan_frame), inplace=True)

            # Update the frequency and phase of this channel.
            channel_trackers[chan].set_frequency_phase(time, frame_freq, frame_phase)

            play = Play(inst.pulse, chan)
            sched.insert(time, play, inplace=True)

        # Insert phase and frequency commands that are ties to physical channels.
        elif isinstance(inst, (SetFrequency, ShiftFrequency)):
            if frames_config[inst.frame].has_physical_channel:
                sched.insert(time, inst, inplace=True)

        elif isinstance(inst, (SetPhase, ShiftPhase)):
            if frames_config[inst.frame].has_physical_channel:
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
