# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Factory of channels.
"""
from .channel_register import ChannelRegister, OutputChannelRegister
from .pulse_channel import PulseChannel
from .backend_channels import OutputChannel
from qiskit.pulse.exceptions import ChannelsError


def create_channel(cls, size: int, name: str = None) -> ChannelRegister:
    if not issubclass(cls, PulseChannel):
        raise ChannelsError("Unknown PulseChannel")

    if issubclass(cls, OutputChannel):
        return OutputChannelRegister(cls, size, name)

    return ChannelRegister(cls, size, name)
