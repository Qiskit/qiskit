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
Special data types.

- InstructionTuple: Type for representing instruction objects to draw.
    Waveform and Frame type instructions are internally converted into this data format and
    fed to the object generators.

- NonPulseTuple: Data set to generate drawing objects.
    Special instructions such as snapshot and relative barriers are internally converted
    into this data format and fed to the object generators.

- ChannelTuple: Data set to generate drawing objects.
    Channel information is internally represented in this data format.

- ComplexColors: Data set to represent a pair of color code associated with a real and
    an imaginary part of filled waveform colors.

- PhaseFreqTuple: Data set to represent a pair of floating values associated with a
    phase and frequency of the frame of channel.
"""

from collections import namedtuple


# Instruction data.
InstructionTuple = namedtuple('InstructionTuple', 't0 dt frame inst')

# Instruction data.
NonPulseTuple = namedtuple('NonPulseTuple', 't0 dt inst')

# Channel information.
ChannelTuple = namedtuple('ChannelTuple', 'channel scaling')

# Color information.
ComplexColors = namedtuple('ComplexColors', 'real imaginary')

# Frame information.
PhaseFreqTuple = namedtuple('PhaseFreqTuple', 'phase freq')
