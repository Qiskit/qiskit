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

# pylint: disable=invalid-name

"""
Special data types.


- PhaseFreqTuple:
    Data set to represent a pair of floating values associated with a
    phase and frequency of the frame of channel.

    phase: Floating value associated with phase.
    freq: Floating value associated with frequency.

- InstructionTuple:
    Type for representing instruction objects to draw.

    Waveform and Frame type instructions are internally converted into this data format and
    fed to the object generators.

    t0: The time when this instruction is issued. This value is in units of cycle time dt.
    dt: Time resolution of this system.
    frame: `PhaseFreqTuple` object to represent the frame of this instruction.
    inst: Pulse instruction object.

- NonPulseTuple:
    Data set to generate drawing objects.

    Special instructions such as snapshot and relative barriers are internally converted
    into this data format and fed to the object generators.

    t0: The time when this instruction is issued. This value is in units of cycle time dt.
    dt: Time resolution of this system.
    inst: Pulse instruction object.

- ChannelTuple:
    Data set to generate drawing objects.

    Channel information is internally represented in this data format.

    channel: Pulse channel object.
    scaling: Vertical scaling factor of the channel.

- ComplexColors:
    Data set to represent a pair of color code associated with a real and
    an imaginary part of filled waveform colors.

    real: Color code for the real part of waveform.
    imaginary: Color code for the imaginary part of waveform.
"""

from enum import Enum
from typing import NamedTuple, Union, List, Optional, NewType

from qiskit import pulse


# TODO: replace with dataclass when py3.5 support is removed.


PhaseFreqTuple = NamedTuple(
    'PhaseFreqTuple',
    [('phase', float),
     ('freq', float)])


InstructionTuple = NamedTuple(
    'InstructionTuple',
    [('t0', int),
     ('dt', Optional[float]),
     ('frame', PhaseFreqTuple),
     ('inst', Union[pulse.Instruction, List[pulse.Instruction]])])


NonPulseTuple = NamedTuple(
    'NonPulseTuple',
    [('t0', int),
     ('dt', Optional[float]),
     ('inst', Union[pulse.Instruction, List[pulse.Instruction]])])


ChannelTuple = NamedTuple(
    'ChannelTuple',
    [('channel', pulse.channels.Channel),
     ('scaling', float)])


ComplexColors = NamedTuple(
    'ComplexColors',
    [('real', str),
     ('imaginary', str)])


ChannelProperty = NamedTuple(
    'ChannelProperty',
    [('scale', float),
     ('visible', bool)])


class DrawingWaveform(str, Enum):
    r"""
    Waveform data type.

    REAL: Assigned to objects that represent real part of waveform.
    IMAG: Assigned to objects that represent imaginary part of waveform.
    """
    REAL = 'Waveform.Real'
    IMAG = 'Waveform.Imag'


class DrawingLabel(str, Enum):
    r"""
    Label data type.

    PULSE_NAME: Assigned to objects that represent name of waveform.
    CH_NAME: Assigned to objects that represent name of channel.
    CH_SCALE: Assigned to objects that represent scaling factor of channel.
    FRAME: Assigned to objects that represent value of frame.
    SNAPSHOT: Assigned to objects that represent label of snapshot.
    """
    PULSE_NAME = 'Label.Pulse.Name'
    CH_NAME = 'Label.Channel.Name'
    CH_SCALE = 'Label.Channel.Scale'
    FRAME = 'Label.Frame.Value'
    SNAPSHOT = 'Label.Snapshot'


class DrawingSymbol(str, Enum):
    r"""
    Symbol data type.

    FRAME: Assigned to objects that represent symbol of frame.
    SNAPSHOT: Assigned to objects that represent symbol of snapshot.
    """
    FRAME = 'Symbol.Frame'
    SNAPSHOT = 'Symbol.Snapshot'


class DrawingLine(str, Enum):
    r"""
    Line data type.

    BASELINE: Assigned to objects that represent zero line of channel.
    BARRIER: Assigned to objects that represent barrier line.
    """
    BASELINE = 'Line.Baseline'
    BARRIER = 'Line.Barrier'


class AbstractCoordinate(str, Enum):
    r"""Abstract coordinate that the exact value depends on the user preference.

    RIGHT: The horizontal coordinate at t0 shifted by the left margin.
    LEFT: The horizontal coordinate at tf shifted by the right margin.
    Y_MAX: The vertical coordinate at the top of the associated channel.
    Y_MIN: The vertical coordinate at the bottom of the associated channel.
    """
    RIGHT = 'RIGHT'
    LEFT = 'LEFT'
    Y_MAX = 'Y_MAX'
    Y_MIN = 'Y_MIN'


class WaveformChannel(pulse.channels.PulseChannel):
    r"""Dummy channel to visualize a waveform."""
    prefix = 'w'

    def __init__(self):
        """Create new waveform channel."""
        super().__init__(0)


Coordinate = NewType('Coordinate', Union[int, float, AbstractCoordinate])
