# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Sample pulse. Deprecated path."""
import warnings

from typing import Optional

from ..channels import PulseChannel
from ..pulse_lib import SamplePulse
from ..instructions import Instruction


class PulseInstruction(Instruction):
    """Instruction to drive a pulse to an `PulseChannel`."""

    def __init__(self, command: SamplePulse, channel: PulseChannel, name: Optional[str] = None):
        warnings.warn("PulseInstruction is deprecated. Use Play, instead, with a pulse and a "
                      "channel. For example: PulseInstruction(SamplePulse([...]), DriveChannel(0))"
                      " -> Play(SamplePulse[...], DriveChannel(0)).",
                      DeprecationWarning)
        super().__init__((), command, (channel,), name=name)
