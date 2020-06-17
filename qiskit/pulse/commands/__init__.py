# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=cyclic-import

"""Supported command types in Pulse. This directory is deprecated.

AcquireInstruction, FrameChange, FrameChangeInstruction, PersistentValue,
PersistentValueInstruction, PulseInstruction, DelayInstruction and ParametricInstruction are all
deprecated. When they are removed, this ``pulse.commands`` import path for the remaining objects
can also be deprecated.
"""
from .acquire import Acquire, AcquireInstruction
from .frame_change import FrameChange, FrameChangeInstruction
from .meas_opts import Discriminator, Kernel
from .persistent_value import PersistentValue, PersistentValueInstruction
from .command import Command
from .pulse_decorators import functional_pulse
from .sample_pulse import SamplePulse, PulseInstruction
from .snapshot import Snapshot
from .delay import Delay, DelayInstruction
from .parametric_pulses import (ParametricPulse, ParametricInstruction, Gaussian, GaussianSquare,
                                Drag, Constant)
