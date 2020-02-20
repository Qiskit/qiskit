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

"""Frame change pulse. Deprecated."""
import warnings

from typing import Optional

from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.instructions import ShiftPhase


def FrameChange(phase: float, name: Optional[str] = None):
    warnings.warn("FrameChange is deprecated. Use ShiftPhase, instead, with channels specified. "
                  "For example: ShiftPhase(3.14)(DriveChannel(0)) -> "
                  "ShiftPhase(3.14, DriveChannel(0)). Returning a ShiftPhase instance. This can "
                  "be called to add a channel: ShiftPhase(3.14)(DriveChannel(0)).",
                  DeprecationWarning)
    return ShiftPhase(phase, name=name)


def FrameChangeInstruction(command: FrameChange, channel: PulseChannel, name=None):
    warnings.warn("The FrameChangeInstruction is deprecated. Use ShiftPhase, instead, with "
                  "channels specified. For example: "
                  "ShiftPhaseInstruction(ShiftPhase(3.14), DriveChannel(0)) -> "
                  "ShiftPhase(3.14, DriveChannel(0)). Returning a ShiftPhase instance.",
                  DeprecationWarning)
    return ShiftPhase(command.phase, channel, name=name)
