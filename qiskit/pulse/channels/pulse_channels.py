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
Channels support signal output.
"""
from abc import ABCMeta

from .channels import Channel


class PulseChannel(Channel, metaclass=ABCMeta):
    """Base class of Channel supporting pulse output."""
    pass


class DriveChannel(PulseChannel):
    """Drive Channel."""

    prefix = 'd'


class MeasureChannel(PulseChannel):
    """Measure Channel."""

    prefix = 'm'


class ControlChannel(PulseChannel):
    """Control Channel."""

    prefix = 'u'
