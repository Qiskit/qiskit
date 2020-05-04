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

"""Delay instruction. Deprecated path."""
import warnings

from ..channels import Channel
from ..instructions import Delay
from ..instructions import Instruction


class DelayInstruction(Instruction):
    """Deprecated."""

    def __init__(self, command: Delay, channel: Channel, name: str = None):
        """Create a delay instruction from a delay command.

        Args:
            command: Delay command to create instruction from.
            channel: Channel for delay instruction to be created.
            name: Name of delay instruction for display purposes.
        """
        warnings.warn("The DelayInstruction is deprecated. Use Delay, instead, with channels. "
                      "For example: DelayInstruction(Delay(5), DriveChannel(0)) -> "
                      "Delay(5, DriveChannel(0)).",
                      DeprecationWarning)
        super().__init__((), command, (channel,), name=name)
