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
"""

from enum import Enum
from typing import NamedTuple, Union, List, Optional, NewType

from qiskit import pulse


PhaseFreqTuple = NamedTuple(
    'PhaseFreqTuple',
    [('phase', float),
     ('freq', float)])
PhaseFreqTuple.__doc__ = 'Data to represent a set of frequency and phase values.'
PhaseFreqTuple.phase.__doc__ = 'Phase value in rad.'
PhaseFreqTuple.freq.__doc__ = 'Frequency value in Hz.'


InstructionTuple = NamedTuple(
    'InstructionTuple',
    [('t0', int),
     ('dt', Optional[float]),
     ('frame', PhaseFreqTuple),
     ('inst', Union[pulse.Instruction, List[pulse.Instruction]])])
InstructionTuple.__doc__ = 'Data to represent pulse instruction for visualization.'
InstructionTuple.t0.__doc__ = 'A time when the instruction is issued.'
InstructionTuple.dt.__doc__ = 'System cycle time.'
InstructionTuple.frame.__doc__ = 'A reference frame to run instruction.'
InstructionTuple.inst.__doc__ = 'Pulse instruction.'


Barrier = NamedTuple(
    'Barrier',
    [('t0', int),
     ('dt', Optional[float]),
     ('channels', List[pulse.channels.Channel])]
)
Barrier.__doc__ = 'Data to represent special pulse instruction of barrier.'
Barrier.t0.__doc__ = 'A time when the instruction is issued.'
Barrier.dt.__doc__ = 'System cycle time.'
Barrier.channels.__doc__ = 'A list of channel associated with this barrier.'


Snapshots = NamedTuple(
    'Snapshots',
    [('t0', int),
     ('dt', Optional[float]),
     ('channels', List[pulse.channels.SnapshotChannel])]
)
Snapshots.__doc__ = 'Data to represent special pulse instruction of snapshot.'
Snapshots.t0.__doc__ = 'A time when the instruction is issued.'
Snapshots.dt.__doc__ = 'System cycle time.'
Snapshots.channels.__doc__ = 'A list of channel associated with this snapshot.'


ChartAxis = NamedTuple(
    'ChartHeader',
    [('name', str)]
)
ChartAxis.__doc__ = 'Data to represent an axis information of chart object'
ChartAxis.name.__doc__ = 'Name of this chart.'


ComplexColors = NamedTuple(
    'ComplexColors',
    [('real', str),
     ('imaginary', str)])
ComplexColors.__doc__ = 'Data to represent a set of color codes for real and imaginary part.'
ComplexColors.real.__doc__ = 'Color code of real part.'
ComplexColors.imaginary.__doc__ = 'Color code of imaginary part.'


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
    """
    RIGHT = 'RIGHT'
    LEFT = 'LEFT'


class WaveformChannel(pulse.channels.PulseChannel):
    r"""Dummy channel that doesn't belong to specific pulse channel."""
    prefix = 'w'

    def __init__(self):
        """Create new waveform channel."""
        super().__init__(0)


# convenient type to represent union of drawing data
DataTypes = NewType('DataType', Union[DrawingWaveform, DrawingLabel, DrawingLine, DrawingSymbol])

# convenient type to represent union of values to represent a coordinate
Coordinate = NewType('Coordinate', Union[int, float, AbstractCoordinate])
