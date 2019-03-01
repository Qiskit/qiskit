# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Factory of channels.
"""
from qiskit.pulse.channels.channel_register import ChannelRegister, OutputChannelRegister
from qiskit.pulse.channels.pulse_channel import PulseChannel
from qiskit.pulse.channels.output_channels import OutputChannel
from qiskit.pulse.exceptions import ChannelsError


def create_channel(cls, size: int, name: str = None) -> ChannelRegister:
    if not issubclass(cls, PulseChannel):
        raise ChannelsError("Unknown PulseChannel")

    if issubclass(cls, OutputChannel):
        return OutputChannelRegister(size, name)

    return ChannelRegister(size, name)

