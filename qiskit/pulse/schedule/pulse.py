# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Pulse = PulseCommand with Channel context.
"""
from qiskit.pulse.commands import PulseCommand
from qiskit.pulse.channels import PulseChannel


class Pulse:
    """MEMO: For scheduler, Pulse = PulseCommand with Channel context."""

    def __init__(self, command: PulseCommand, channel: PulseChannel):
        self.command = command
        self.channel = channel


