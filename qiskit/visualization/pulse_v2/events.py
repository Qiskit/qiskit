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
"""
Channel event manager for pulse schedules.

This

"""
from collections import defaultdict, namedtuple, OrderedDict

from typing import Tuple, Union, Optional, Dict, List

from qiskit import pulse
from qiskit.visualization.exceptions import VisualizationError
import numpy as np


Waveform = namedtuple('Waveform', 'samples phase frequency attribute')
Frame = namedtuple('Frame', 'type value')
Auxiliary = namedtuple('Auxiliary', 'type attribute')


class ChannelEvents:
    """Channel event manager.
    """

    def __init__(self,
                 waveforms: Dict[int, Waveform],
                 frames: Dict[int, List[Waveform]],
                 snapshots: Dict[int, List[Auxiliary]],
                 channel: pulse.channels.Channel):
        """Create new event manager.

        Args:
            waveforms: List of waveforms shown in this channel.
            frames: List of frame instructions shown in this channel.
            snapshots: List of snapshot instruction shown in this channel.
            channel: Channel object associated with this manager.
        """
        self.waveforms = waveforms
        self.frames = frames
        self.snapshots = snapshots
        self.channel = channel

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

            # parse frame
            filter_kwargs = {
                'channels': [channel],
                'instruction_types': [pulse.SetPhase,
                                      pulse.ShiftPhase,
                                      pulse.SetFrequency,
                                      pulse.ShiftFrequency]
            }
            phase, freq = {0: 0}, {0: 0}
            for t0, inst in program.filter(**filter_kwargs).instructions:
                if isinstance(inst, pulse.SetPhase):
                    oper = Frame(type=type(inst), value=inst.phase)
                    phase = _update_frame(t0, inst.phase, phase, overwrite=True)
                elif isinstance(inst, pulse.ShiftPhase):
                    oper = Frame(type=type(inst), value=inst.phase)
                    phase = _update_frame(t0, inst.phase, phase, overwrite=False)
                elif isinstance(inst, pulse.SetFrequency):
                    oper = Frame(type=type(inst), value=inst.frequency)
                    freq = _update_frame(t0, inst.frequency, freq, overwrite=True)
                elif isinstance(inst, pulse.ShiftFrequency):
                    oper = Frame(type=type(inst), value=inst.frequency)
                    freq = _update_frame(t0, inst.frequency, freq, overwrite=False)
                else:
                    continue
                frames[t0].append(oper)

            # parse other instructions
            for t0, inst in program.filter(channels=[channel]).instructions:
                if isinstance(inst, pulse.Play):
                    # Pulse type
                    target = inst.pulse
                    attribute = {
                        'name': target.name,
                        'id': target.id,
                        'duration': target.duration
                    }
                    # convert ParametricPulse into SamplePulse
                    if isinstance(target, pulse.ParametricPulse):
                        target = target.get_sample_pulse()
                        attribute.update(**target.parameters)
                    oper = Waveform(samples=target.samples,
                                    phase=_get_nearlest_frame(t0, phase),
                                    frequency=_get_nearlest_frame(t0, freq),
                                    attribute=attribute)
                    waveforms[t0] = oper
                elif isinstance(inst, pulse.Delay):
                    # Delay type
                    attribute = {
                        'duration': inst.duration,
                        'name': 'Delay'
                    }
                    oper = Waveform(samples=np.zeros(inst.duration),
                                    phase=0,
                                    frequency=0,
                                    attribute=attribute)
                    waveforms[t0] = oper
                elif isinstance(inst, pulse.Acquire):
                    # Acquire type
                    attribute = {
                        'kernel': inst.kernel,
                        'discriminator': inst.discriminator,
                        'mem_slot': inst.mem_slot,
                        'reg_slot': inst.reg_slot,
                        'duration': inst.duration,
                        'name': 'Acquire'
                    }
                    oper = Waveform(samples=np.ones(inst.duration),
                                    phase=0,
                                    frequency=0,
                                    attribute=attribute)
                    waveforms[t0] = oper
                elif isinstance(inst, pulse.Snapshot):
                    # Snapshot type
                    attribute = {
                        'label': inst.label,
                        'snapshot_type': inst.type,
                        'name': 'Snapshot'
                    }
                    oper = Auxiliary(type=type(inst), attribute=attribute)
                    snapshots[t0].append(oper)

        elif isinstance(program, pulse.SamplePulse):
            # sample pulse
            attribute = {
                'name': program.name,
                'id': program.id,
                'duration': program.duration
            }
            oper = Waveform(samples=program.samples,
                            phase=0,
                            frequency=0,
                            attribute=attribute)
            waveforms[0] = oper
        else:
            VisualizationError('%s is not valid data type for this drawer.' % type(program))

        return ChannelEvents(waveforms, frames, snapshots, channel)

    def is_empty(self):
        """Check if there is any nonzero waveforms in this channel.

        Note:
            Frame and other auxiliary type instructions are ignored.
        """
        for waveform in self.waveforms.values():
            if np.nonzero(waveform.samples)[0].size > 0:
                return False
        else:
            return True


def _update_frame(t0: int,
                  value: float,
                  frame: Dict[int, float],
                  overwrite: bool) -> Dict[int, float]:
    """Update frame value.

    Args:
        t0: Issued time of frame change instruction.
        value: New frame value.
        frame: List of phase or frequency with time.
        overwrite: Set ``True`` when frame value is overwritten.
    """
    if not overwrite:
        if t0 not in frame:
            frame[t0] = frame[max(frame.keys())] + value
        else:
            frame[t0] += value
    else:
        frame[t0] = value

    return frame


def _get_nearlest_frame(t0: int,
                        frames: Dict[int, float]) -> float:
    """Find frame value at time t0.

    Args:
        t0: Issued time of instruction.
        frames: List of phase or frequency with time.
    """
    time_array = np.array(sorted(list(frames.keys())))
    ind = np.argwhere(time_array <= t0)[-1]

    return frames[time_array[ind[0]]]
