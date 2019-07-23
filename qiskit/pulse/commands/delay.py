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

"""
Delay instruction.
"""
from typing import Optional

from qiskit.pulse.channels import Channel
from .instruction import Instruction
from .command import Command


class Delay(Command):
    """Delay command."""

    prefix = 'delay'

    def __init__(self, duration: int, name: Optional[str] = None):
        """A delay command.

        No other may be scheduled within a delay command.

        Args:
            duration: int
            name: Name of delay command for display purposes.
        """
        super().__init__(duration=duration)
        self._name = Delay.create_name(name)

    # pylint: disable=arguments-differ
    def to_instruction(self, channel: Channel, name=None) -> 'DelayInstruction':
        return DelayInstruction(self, channel, name=name)
    # pylint: enable=arguments-differ


class DelayInstruction(Instruction):
    """Instruction to add a blocking delay on a `Channel`.

    No other may be scheduled during a delay.
    """

    def __init__(self, command: Delay, channel: Delay, name=None):
        super().__init__(command, channel, name=name)
