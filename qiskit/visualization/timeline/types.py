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
Special data types.

- ScheduledGate:
    t0: Time when the instruction is issued.
    operand: Gate object associated with the instruction.
    duration: Time duration of the instruction.
    bits:

- GateLink:
    t0: Position where the link is placed.
    operand: Gate object associated with the instruction.
    bits: List of bit associated with the instruction.
"""

from enum import Enum
from typing import NamedTuple, List, Union

from qiskit import circuit


ScheduledGate = NamedTuple(
    'ScheduledGate',
    [('t0', int),
     ('operand', circuit.Gate),
     ('duration', int),
     ('bits', List[Union[circuit.Qubit, circuit.Clbit]])])

GateLink = NamedTuple(
    'GateLink',
    [('t0', int),
     ('operand', circuit.Gate),
     ('bits', List[Union[circuit.Qubit, circuit.Clbit]])])


class DrawingBox(str, Enum):
    r"""Box data type.

    SCHED_GATE: Box that represents occupation time by gate.
    TIME_LINE: Box that represents time slot of a bit.
    """
    SCHED_GATE = 'Box.ScheduledGate'
    TIME_LINE = 'Box.Timeline'


class DrawingLine(str, Enum):
    r"""Line data type.

    BARRIER: Line that represents barrier instruction.
    """
    BARRIER = 'Line.Barrier'


class DrawingSymbol(str, Enum):
    r"""Symbol data type.

    FRAME: Symbol that represents zero time frame change (Rz) instruction.
    """
    FRAME = 'Symbol.Frame'


class DrawingLabel(str, Enum):
    r"""Label data type.

    GATE_NAME: Label that represents name of gate.
    GATE_PARAM: Label that represents parameter of gate.
    BIT_NAME: Label that represents name of bit.
    """
    GATE_NAME = 'Label.Gate.Name'
    GATE_PARAM = 'Label.Gate.Param'
    BIT_NAME = 'Label.Bit.Name'
