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
"""

from typing import NamedTuple, Union, List

from qiskit import pulse


class PhaseFreqTuple(NamedTuple):
    """Data set to represent a pair of floating values associated with a
    phase and frequency of the frame of channel.
    """
    phase: float
    freq: float


class InstructionTuple(NamedTuple):
    """Type for representing instruction objects to draw.

    Waveform and Frame type instructions are internally converted into this data format and
    fed to the object generators.
    """
    t0: int
    dt: float
    frame: PhaseFreqTuple
    inst: Union[pulse.Instruction, List[pulse.Instruction]]


class NonPulseTuple(NamedTuple):
    """Data set to generate drawing objects.

    Special instructions such as snapshot and relative barriers are internally converted
    into this data format and fed to the object generators.
    """
    t0: int
    dt: float
    inst: Union[pulse.Instruction, List[pulse.Instruction]]


class ChannelTuple(NamedTuple):
    """Data set to generate drawing objects.

    Channel information is internally represented in this data format.
    """
    channel: pulse.channels.Channel
    scaling: float


class ComplexColors(NamedTuple):
    """Data set to represent a pair of color code associated with a real and
    an imaginary part of filled waveform colors.
    """
    real: str
    imag: str
