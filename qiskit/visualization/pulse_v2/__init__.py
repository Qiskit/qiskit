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
Pulse visualization module.
"""

from collections import namedtuple
from qiskit.visualization.pulse_v2.stylesheet import QiskitPulseStyle, hard_corded_default_style
from qiskit.visualization.pulse_v2 import generators

# custom data types
InstructionTuple = namedtuple('InstructionTuple', 't0 dt frame inst')
NonPulseTuple = namedtuple('NonPulseTuple', 't0 dt inst')
ChannelTuple = namedtuple('ChannelTuple', 'channel scaling t0 t1')
ComplexColors = namedtuple('ComplexColors', 'real imaginary')


# create default stylesheet
pulse_style = QiskitPulseStyle()
pulse_style.style = hard_corded_default_style
