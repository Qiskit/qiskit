# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Pulse Channels.
"""
from abc import ABCMeta, abstractmethod

from qiskit.pulse import commands


class PulseChannel(metaclass=ABCMeta):
    """Pulse Channel."""

    supported = commands.PulseCommand

    @abstractmethod
    def name(self):
        pass
