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
from typing import NamedTuple, Union, List, Optional, NewType, Dict, Any, Tuple

import numpy as np
from qiskit import pulse

PhaseFreqTuple = NamedTuple(
    'PhaseFreqTuple',
    [('phase', float),
     ('freq', float)])
PhaseFreqTuple.__doc__ = 'Data to represent a set of frequency and phase values.'
PhaseFreqTuple.phase.__doc__ = 'Phase value in rad.'
PhaseFreqTuple.freq.__doc__ = 'Frequency value in Hz.'


PulseInstruction = NamedTuple(
    'InstructionTuple',
    [('t0', int),
     ('dt', Optional[float]),
     ('frame', PhaseFreqTuple),
     ('inst', Union[pulse.Instruction, List[pulse.Instruction]])])
PulseInstruction.__doc__ = 'Data to represent pulse instruction for visualization.'
PulseInstruction.t0.__doc__ = 'A time when the instruction is issued.'
PulseInstruction.dt.__doc__ = 'System cycle time.'
PulseInstruction.frame.__doc__ = 'A reference frame to run instruction.'
PulseInstruction.inst.__doc__ = 'Pulse instruction.'


BarrierInstruction = NamedTuple(
    'Barrier',
    [('t0', int),
     ('dt', Optional[float]),
     ('channels', List[pulse.channels.Channel])]
)
BarrierInstruction.__doc__ = 'Data to represent special pulse instruction of barrier.'
BarrierInstruction.t0.__doc__ = 'A time when the instruction is issued.'
BarrierInstruction.dt.__doc__ = 'System cycle time.'
BarrierInstruction.channels.__doc__ = 'A list of channel associated with this barrier.'


SnapshotInstruction = NamedTuple(
    'Snapshots',
    [('t0', int),
     ('dt', Optional[float]),
     ('inst', pulse.instructions.Snapshot)]
)
SnapshotInstruction.__doc__ = 'Data to represent special pulse instruction of snapshot.'
SnapshotInstruction.t0.__doc__ = 'A time when the instruction is issued.'
SnapshotInstruction.dt.__doc__ = 'System cycle time.'
SnapshotInstruction.inst.__doc__ = 'Snapshot instruction.'


ChartAxis = NamedTuple(
    'ChartAxis',
    [('name', str),
     ('channels', List[pulse.channels.Channel])]
)
ChartAxis.__doc__ = 'Data to represent an axis information of chart.'
ChartAxis.name.__doc__ = 'Name of chart.'
ChartAxis.channels.__doc__ = 'Channels associated with chart.'


ComplexColors = NamedTuple(
    'ComplexColors',
    [('real', str),
     ('imaginary', str)])
ComplexColors.__doc__ = 'Data to represent a set of color codes for real and imaginary part.'
ComplexColors.real.__doc__ = 'Color code of real part.'
ComplexColors.imaginary.__doc__ = 'Color code of imaginary part.'


ParsedInstruction = NamedTuple(
    'ParsedWaveform',
    [('xvals', np.ndarray),
     ('yvals', np.ndarray),
     ('meta', Dict[str, Any])]
)
ParsedInstruction.__doc__ = 'Data to represent a parsed pulse instruction for object generation.'
ParsedInstruction.xvals.__doc__ = 'Numpy array of x axis data.'
ParsedInstruction.yvals.__doc__ = 'Numpy array of y axis data.'
ParsedInstruction.meta.__doc__ = 'Dictionary containing instruction details.'


HorizontalAxis = NamedTuple(
    'HorizontalAxis',
    [('window', Tuple[int, int]),
     ('axis_map', Dict[int, int]),
     ('axis_break_pos', List[int]),
     ('label', str)]
)
HorizontalAxis.__doc__ = "Data to represent configuration of horizontal axis."
HorizontalAxis.window.__doc__ = "Left and right edge of graph."
HorizontalAxis.axis_map.__doc__ = 'Mapping of apparent coordinate system and actual location.'
HorizontalAxis.axis_break_pos.__doc__ = "Locations of axis break."
HorizontalAxis.label.__doc__ = "Label of horizontal axis."


class DrawingWaveform(Enum):
    """
    Waveform data type.

    REAL: Assigned to objects that represent real part of waveform.
    IMAG: Assigned to objects that represent imaginary part of waveform.
    """
    REAL = 'Waveform.Real'
    IMAG = 'Waveform.Imag'


class DrawingLabel(Enum):
    """
    Label data type.

    PULSE_NAME: Assigned to objects that represent name of waveform.
    CH_NAME: Assigned to objects that represent name of channel.
    CH_SCALE: Assigned to objects that represent scaling factor of channel.
    FRAME: Assigned to objects that represent value of frame.
    SNAPSHOT: Assigned to objects that represent label of snapshot.
    """
    PULSE_NAME = 'Label.Pulse.Name'
    PULSE_INFO = 'Label.Pulse.Info'
    CH_NAME = 'Label.Channel.Name'
    CH_INFO = 'Label.Channel.Info'
    FRAME = 'Label.Frame.Value'
    SNAPSHOT = 'Label.Snapshot'


class DrawingSymbol(Enum):
    """
    Symbol data type.

    FRAME: Assigned to objects that represent symbol of frame.
    SNAPSHOT: Assigned to objects that represent symbol of snapshot.
    """
    FRAME = 'Symbol.Frame'
    SNAPSHOT = 'Symbol.Snapshot'


class DrawingLine(Enum):
    """
    Line data type.

    BASELINE: Assigned to objects that represent zero line of channel.
    BARRIER: Assigned to objects that represent barrier line.
    """
    BASELINE = 'Line.Baseline'
    BARRIER = 'Line.Barrier'


class AbstractCoordinate(Enum):
    """Abstract coordinate that the exact value depends on the user preference.

    RIGHT: The horizontal coordinate at t0 shifted by the left margin.
    LEFT: The horizontal coordinate at tf shifted by the right margin.
    TOP: The vertical coordinate at the top of chart.
    BOTTOM: The vertical coordinate at the bottom of chart.
    """
    RIGHT = 'RIGHT'
    LEFT = 'LEFT'
    TOP = 'TOP'
    BOTTOM = 'BOTTOM'


class DynamicString(str, Enum):
    """The string which is dynamically updated at the time of drawing.

    SCALE: A temporal value of chart scaling factor.
    """
    SCALE = '@scale'


class WaveformChannel(pulse.channels.PulseChannel):
    """Dummy channel that doesn't belong to specific pulse channel."""
    prefix = 'w'

    def __init__(self):
        """Create new waveform channel."""
        super().__init__(0)


# convenient type to represent union of drawing data
DataTypes = NewType('DataType', Union[DrawingWaveform, DrawingLabel, DrawingLine, DrawingSymbol])

# convenient type to represent union of values to represent a coordinate
Coordinate = NewType('Coordinate', Union[int, float, AbstractCoordinate])
