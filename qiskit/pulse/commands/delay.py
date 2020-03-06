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
        """Create a new delay command.

        No other command may be scheduled within a delay command.

        Delays that exist as the last instruction on a channel will be removed.

        Note: Delays are not explicitly sent to backends with the current Qobj
        transport layer. They are only used to enforce the correct offset in time
        for instructions that will be submitted directly to the backend.

        Args:
            duration: Length of time of the delay in terms of dt.
            name: Name of delay command for display purposes.
        """
        super().__init__(duration=duration)
        self._name = Delay.create_name(name)

    # pylint: disable=arguments-differ
    def to_instruction(self, channel: Channel, name: str = None) -> 'DelayInstruction':
        """Create a delay instruction on the given channel.

        Args:
            channel: Channel for delay instruction to be created.
            name: Name of delay instruction for display purposes.

        Returns:
            DelayInstruction
        """
        return DelayInstruction(self, channel, name=name)
    # pylint: enable=arguments-differ


class DelayInstruction(Instruction):
    """Instruction to add a blocking delay on a `Channel`.

    No other may be scheduled within a delay command.

    Delays that exist as the last instruction on a channel will be removed.

    Note: Delay's are not explicitly sent to backends with the current Qobj
    transport layer. They are only used to enforce the correct offset in time
    for instructions that will be submitted directly to the backend.
    """

    def __init__(self, command: Delay, channel: Delay, name: str = None):
        """Create a delay instruction from a delay command.

        Args:
            command: Delay command to create instruction from.
            channel: Channel for delay instruction to be created.
            name: Name of delay instruction for display purposes.
        """
        super().__init__(command, channel, name=name)
