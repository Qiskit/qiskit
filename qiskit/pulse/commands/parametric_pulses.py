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

"""Deprecated path to parametric pulses."""
import warnings

# pylint: disable=unused-import

from qiskit.pulse.pulse_lib import (ParametricPulse, Gaussian, GaussianSquare,
                                    Drag, Constant, ConstantPulse)
from qiskit.pulse.channels import Channel


class ParametricInstruction:
    """Instruction to drive a parametric pulse to an `PulseChannel`."""

    def __init__(self, command: ParametricPulse, channel: Channel, name: str = None):
        warnings.warn("ParametricInstruction is deprecated. Use Play, instead, with a pulse and a "
                      "channel. For example: ParametricInstruction(Gaussian(amp=amp, sigma=sigma, "
                      "duration=duration), DriveChannel(0)) -> Play(Gaussian(amp=amp, sigma=sigma,"
                      " duration=duration), DriveChannel(0)).",
                      DeprecationWarning)
        super().__init__((), command, (channel,), name=name)
