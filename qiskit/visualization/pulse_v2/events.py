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

r"""
Channel event manager for pulse schedules.

This module provides a `ChannelEvents` class that manages a series of instructions for a
pulse channel. Channel-wise filtering of the pulse program makes
the arrangement of channels easier in the core drawer function.
The `ChannelEvents` class is expected to be called by other programs (not by end-users).

The `ChannelEvents` class instance is created with the class method ``load_program``:
    ```python
    event = ChannelEvents.load_program(sched, DriveChannel(0))
    ```

The `ChannelEvents` is created for a specific pulse channel and loosely assorts pulse
instructions within the channel with different visualization purposes.

Phase and frequency related instructions are loosely grouped as frame changes.
The instantaneous value of those operands are combined and provided as ``PhaseFreqTuple``.
Instructions that have finite duration are grouped as waveforms.

The grouped instructions are returned as an iterator by the corresponding method call:
    ```python
    for t0, frame, instruction in event.get_waveforms():
        ...

    for t0, frame_change, instructions in event.get_frame_changes():
        ...
    ```

The class method ``get_waveforms`` returns the iterator of waveform type instructions with
the ``PhaseFreqTuple`` (frame) at the time when instruction is issued.
This is because a pulse envelope of ``Waveform`` may be modulated with a
phase factor $exp(-i \omega t - \phi)$ with frequency $\omega$ and phase $\phi$ and
appear on the canvas. Thus, it is better to tell users in which phase and frequency
the pulse envelope is modulated from a viewpoint of program debugging.

On the other hand, the class method ``get_frame_changes`` returns a ``PhaseFreqTuple`` that
represents a total amount of change at that time because it is convenient to know
the operand value itself when we debug a program.

Because frame change type instructions are usually zero duration, multiple instructions
can be issued at the same time and those operand values should be appropriately
combined. In Qiskit Pulse we have set and shift type instructions for the frame control,
the set type instruction will be converted into the relevant shift amount for visualization.
Note that these instructions are not interchangeable and the order should be kept.
For example:
    ```python
    sched1 = Schedule()
    sched1 = sched1.insert(0, ShiftPhase(-1.57, DriveChannel(0))
    sched1 = sched1.insert(0, SetPhase(3.14, DriveChannel(0))

    sched2 = Schedule()
    sched2 = sched2.insert(0, SetPhase(3.14, DriveChannel(0))
    sched2 = sched2.insert(0, ShiftPhase(-1.57, DriveChannel(0))
    ```
In this example, ``sched1`` and ``sched2`` will have different frames.
On the drawer canvas, the total frame change amount of +3.14 should be shown for ``sched1``,
while `sched2` is +1.57. Since the `SetPhase` and the `ShiftPhase` instruction behave
differently, we cannot simply sum up the operand values in visualization output.

It should be also noted that zero duration instructions issued at the same time will be
overlapped on the canvas. Thus it is convenient to plot a total frame change amount rather
than plotting each operand value bound to the instruction.
"""
from collections import defaultdict
from typing import Dict, List, Iterator, Tuple

from qiskit import pulse
from qiskit.visualization.pulse_v2.types import PhaseFreqTuple


class ChannelEvents:
    """Channel event manager.
    """
    _waveform_group = tuple((pulse.instructions.Play,
                             pulse.instructions.Delay,
                             pulse.instructions.Acquire))
    _frame_group = tuple((pulse.instructions.SetFrequency,
                          pulse.instructions.ShiftFrequency,
                          pulse.instructions.SetPhase,
                          pulse.instructions.ShiftPhase))

    def __init__(self,
                 waveforms: Dict[int, pulse.Instruction],
                 frames: Dict[int, List[pulse.Instruction]],
                 channel: pulse.channels.Channel):
        """Create new event manager.

        Args:
            waveforms: List of waveforms shown in this channel.
            frames: List of frame change type instructions shown in this channel.
            channel: Channel object associated with this manager.
        """
        self._waveforms = waveforms
        self._frames = frames
        self.channel = channel

        # initial frame
        self.init_phase = 0
        self.init_frequency = 0

    @classmethod
    def load_program(cls,
                     program: pulse.Schedule,
                     channel: pulse.channels.Channel):
        """Load a pulse program represented by ``Schedule``.

        Args:
            program: Target ``Schedule`` to visualize.
            channel: The channel managed by this instance.

        Returns:
            ChannelEvents: The channel event manager for the specified channel.
        """
        waveforms = dict()
        frames = defaultdict(list)

        # parse instructions
        for t0, inst in program.filter(channels=[channel]).instructions:
            if isinstance(inst, cls._waveform_group):
                waveforms[t0] = inst
            elif isinstance(inst, cls._frame_group):
                frames[t0].append(inst)

        return ChannelEvents(waveforms, frames, channel)

    def is_empty(self):
        """Check if there is any nonzero waveforms in this channel."""
        for waveform in self._waveforms.values():
            if isinstance(waveform, (pulse.instructions.Play, pulse.instructions.Acquire)):
                return False
        return True

    def get_waveforms(self) -> Iterator[Tuple[int, PhaseFreqTuple, pulse.Instruction]]:
        """Return waveform type instructions with frame."""
        sorted_frame_changes = sorted(self._frames.items(), key=lambda x: x[0], reverse=True)
        sorted_waveforms = sorted(self._waveforms.items(), key=lambda x: x[0])

        # bind phase and frequency with instruction
        phase = self.init_phase
        frequency = self.init_frequency
        for t0, inst in sorted_waveforms:
            while len(sorted_frame_changes) > 0 and sorted_frame_changes[-1][0] <= t0:
                _, frame_changes = sorted_frame_changes.pop()
                for frame_change in frame_changes:
                    if isinstance(frame_change, pulse.instructions.SetFrequency):
                        frequency = frame_change.frequency
                    elif isinstance(frame_change, pulse.instructions.ShiftFrequency):
                        frequency += frame_change.frequency
                    elif isinstance(frame_change, pulse.instructions.SetPhase):
                        phase = frame_change.phase
                    elif isinstance(frame_change, pulse.instructions.ShiftPhase):
                        phase += frame_change.phase
            frame = PhaseFreqTuple(phase, frequency)

            yield t0, frame, inst

    def get_frame_changes(self) -> Iterator[Tuple[int, PhaseFreqTuple, List[pulse.Instruction]]]:
        """Return frame change type instructions with total frame change amount."""
        sorted_frame_changes = sorted(self._frames.items(), key=lambda x: x[0])

        phase = self.init_phase
        frequency = self.init_frequency
        for t0, insts in sorted_frame_changes:
            pre_phase = phase
            pre_frequency = frequency
            for inst in insts:
                if isinstance(inst, pulse.instructions.SetFrequency):
                    frequency = inst.frequency
                elif isinstance(inst, pulse.instructions.ShiftFrequency):
                    frequency += inst.frequency
                elif isinstance(inst, pulse.instructions.SetPhase):
                    phase = inst.phase
                elif isinstance(inst, pulse.instructions.ShiftPhase):
                    phase += inst.phase
            frame = PhaseFreqTuple(phase - pre_phase, frequency - pre_frequency)

            yield t0, frame, insts
