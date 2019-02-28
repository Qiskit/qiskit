# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Factory of channels.
"""
from typing import List
from qiskit.pulse.channels.pulse_channel import PulseChannel
from qiskit.pulse.channels.output_channels import OutputChannel
from qiskit.pulse.exceptions import ChannelsError


def create_channel(classinfo, size: int, lo_frequencies: List[float] = None) -> List:
    if not issubclass(classinfo, PulseChannel):
        raise ChannelsError("Unknown PulseChannel")

    if issubclass(classinfo, OutputChannel):
        if lo_frequencies is not None:
            if len(lo_frequencies) == size:
                return [classinfo(i, lof) for i, lof in zip(range(size), lo_frequencies)]
            else:
                raise ChannelsError("the size of lo_frequencies must be size")
    else:
        if lo_frequencies is not None:
            raise ChannelsError("cannot apply lo_frequencies to this type of channel")

    return [classinfo(i) for i in range(size)]
