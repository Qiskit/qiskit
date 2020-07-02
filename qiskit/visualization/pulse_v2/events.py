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

This module provide `ChannelEvents` class that manages series of instruction in
specified channel. This makes arrangement of pulse channels easier later in
the core drawing function. The `ChannelEvents` class is expected to be called by
other programs, not by end-users.

The `ChannelEvents` class instance is created with the class method ``parse_program``:
    ```python
    event = ChannelEvents.parse_program(sched, DriveChannel(0))
    ```

The manager is created for a specific pulse channel and assorts pulse instructions within
the channel with different visualization types.
A phase and frequency related instruction are managed as frame change.
Those information is combined as ``PhaseFreqTuple``.

The grouped instructions are returned as an iterator by corresponding method call:
    ```python
    for t0, frame, instruction in event.get_waveform():
        ...

    for t0, frame_change, instructions in event.get_framechange():
        ...
    ```

The classmethod ``get_waveform`` returns the frame at the time when instruction is issued.
This is because a pulse envelope of ``SamplePulse`` may be modulated with a phase factor
$exp(-i \omega t - \phi)$ with frequency $\omega$ and phase $\phi$ in the canvas of the plotter.
Thus, it is better to tell users in which phase and frequency the pulse is modulated
from a viewpoint of debugging.

On the other hand, the classmethod ``get_framechange`` returns an amount of change of the frame
at the time when instructions are issued. Because we have set and shift type instructions,
the set type instruction will be converted into the corresponding shift amount for visualization.

Because a frame change instruction is zero duration, multiple instructions can be issued
at the same time and those instructions are order sensitive.
For example:
    ```python
    sched1 = Schedule()
    sched1 = sched1.insert(0, ShiftPhase(-1.57, DriveChannel(0))
    sched1 = sched1.insert(0, SetPhase(3.14, DriveChannel(0))

    sched2 = Schedule()
    sched2 = sched2.insert(0, SetPhase(3.14, DriveChannel(0))
    sched2 = sched2.insert(0, ShiftPhase(-1.57, DriveChannel(0))
    ```
In above example, `sched1` and `sched2` will create different frame.
The total frame change value of `sched1` should be +3.14, while `sched2` is +1.57.
Since the `SetPhase` and the `ShiftPhase` instruction behave differently,
we cannot simply sum up the phase values to show.

It should be noted that zero-time instructions issued at the same time will be overlapped on the
canvas of the plotter. Thus it is convenient to plot an apparent frame change value rather
than plotting each phase value bound to the instruction.
"""
from collections import defaultdict, namedtuple
from typing import Union, Optional, Dict, List, Iterator, Tuple

from qiskit import pulse
from qiskit.visualization.exceptions import VisualizationError

PhaseFreqTuple = namedtuple('PhaseFreqTuple', 'phase freq')


class ChannelEvents:
    """Channel event manager.
    """

    def __init__(self,
                 waveforms: Dict[int, pulse.Instruction],
                 frames: Dict[int, List[pulse.Instruction]],
                 snapshots: Dict[int, List[pulse.Instruction]],
                 channel: pulse.channels.Channel):
        """Create new event manager.

        Args:
            waveforms: List of waveforms shown in this channel.
            frames: List of frame change type instructions shown in this channel.
            snapshots: List of snapshot instruction shown in this channel.
            channel: Channel object associated with this manager.
        """
        self._waveforms = waveforms
        self._frames = frames
        self._snapshots = snapshots
        self.channel = channel

        # initial frame
        self.init_phase = 0
        self.init_frequency = 0

    @classmethod
    def parse_program(cls,
                      program: Union[pulse.Schedule, pulse.SamplePulse],
                      channel: Optional[pulse.channels.Channel] = None):
        """Parse pulse program represented by ``Schedule`` or ``SamplePulse``.

        Args:
            program: Target program to visualize, either ``Schedule`` or ``SamplePulse``.
            channel: Target channel of managed by this instance.
                This information is necessary when program is ``Schedule``.

        Raises:
            VisualizationError: When ``Schedule`` is given and ``channel`` is empty.
            VisualizationError: When invalid data is specified.
        """
        waveforms = dict()
        frames = defaultdict(list)
        snapshots = defaultdict(list)

        if isinstance(program, pulse.Schedule):
            # schedule
            if channel is None:
                raise VisualizationError('Pulse channel should be specified.')

            # parse instructions
            for t0, inst in program.filter(channels=[channel]).instructions:
                if isinstance(inst, pulse.Play):
                    # play
                    waveforms[t0] = inst.pulse
                elif isinstance(inst, (pulse.Delay, pulse.Acquire)):
                    # waveform type
                    waveforms[t0] = inst
                elif isinstance(inst, (pulse.SetFrequency, pulse.ShiftFrequency,
                                       pulse.SetPhase, pulse.ShiftPhase)):
                    # frame types
                    frames[t0].append(inst)
                elif isinstance(inst, pulse.Snapshot):
                    # Snapshot type
                    snapshots[t0].append(inst)

        elif isinstance(program, pulse.SamplePulse):
            # sample pulse
            waveforms[0] = program
        else:
            VisualizationError('%s is not valid data type for this drawer.' % type(program))

        return ChannelEvents(waveforms, frames, snapshots, channel)

    def is_empty(self):
        """Check if there is any nonzero waveforms in this channel.

        Note:
            Frame and other auxiliary type instructions are ignored.
        """
        for waveform in self._waveforms.values():
            if isinstance(waveform, (pulse.SamplePulse, pulse.ParametricPulse, pulse.Acquire)):
                return False
        else:
            return True

    def get_waveform(self) -> Iterator[Tuple[int, PhaseFreqTuple, pulse.Instruction]]:
        """Return waveform type instructions with phase and frequency.
        """
        sorted_frame_changes = sorted(self._frames.items(), key=lambda x: x[0], reverse=True)

        # bind phase and frequency with instruction
        phase = self.init_phase
        frequency = self.init_frequency
        for t0, inst in self._waveforms.items():
            while len(sorted_frame_changes) > 0 and sorted_frame_changes[-1][0] <= t0:
                _, frame_changes = sorted_frame_changes.pop()
                for frame_change in frame_changes:
                    if isinstance(frame_change, pulse.SetFrequency):
                        frequency = frame_change.frequency
                    elif isinstance(frame_change, pulse.ShiftFrequency):
                        frequency += frame_change.frequency
                    elif isinstance(frame_change, pulse.SetPhase):
                        phase = frame_change.phase
                    elif isinstance(frame_change, pulse.ShiftPhase):
                        phase += frame_change.phase
            frame = PhaseFreqTuple(phase, frequency)

            yield t0, frame, inst

    def get_framechange(self) -> Iterator[Tuple[int, PhaseFreqTuple, List[pulse.Instruction]]]:
        """Return frame change type instructions with total phase and total frequency.
        """
        sorted_frame_changes = sorted(self._frames.items(), key=lambda x: x[0])

        phase = self.init_phase
        frequency = self.init_frequency
        for t0, insts in sorted_frame_changes:
            pre_phase = phase
            pre_frequency = frequency
            for inst in insts:
                if isinstance(inst, pulse.SetFrequency):
                    frequency = inst.frequency
                elif isinstance(inst, pulse.ShiftFrequency):
                    frequency += inst.frequency
                elif isinstance(inst, pulse.SetPhase):
                    phase = inst.phase
                elif isinstance(inst, pulse.ShiftPhase):
                    phase += inst.phase
            frame = PhaseFreqTuple(phase - pre_phase, frequency - pre_frequency)

            yield t0, frame, insts

    def get_snapshots(self) -> Iterator[Tuple[int, List[pulse.Instruction]]]:
        """Return snapshot instructions.
        """
        for t0, insts in self._snapshots:
            yield t0, insts
