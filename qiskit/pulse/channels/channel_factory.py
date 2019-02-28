# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Factory of channels.
"""
from typing import List
from qiskit.pulse.exceptions import ChannelsError
from qiskit.pulse.channels.output_channels import DriveChannel, ControlChannel, MeasureChannel


def create_channel(classinfo, size: int) -> List:
    return [classinfo(i) for i in range(size)]


def create_output_channel(classinfo, size: int, lo_frequencies: List[float] = None) -> List:
    if type(classinfo) not in [DriveChannel, ControlChannel, MeasureChannel]:
        raise ChannelsError("%s is not OutputChannel, it cannot be created." % str(classinfo))
    if lo_frequencies:
        if len(lo_frequencies) != size:
            raise ChannelsError("the size of lo_frequencies must be size")
        return [classinfo(i, lof) for i, lof in zip(range(size), lo_frequencies)]
    return [classinfo(i) for i in range(size)]
