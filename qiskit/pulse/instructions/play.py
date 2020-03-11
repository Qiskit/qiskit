# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""TODO"""
from typing import Optional

from qiskit.pulse.channels import PulseChannel

from .instruction import Instruction


class Play(Instruction):
    """TODO"""

    def __init__(self, pulse: Pulse, channel: PulseChannel, name: Optional[str] = None):
        """TODO"""
        self._pulse = pulse
        self._channel = channel
        super().__init__(pulse.duration, channel, name=name)

    @property
    def operands(self) -> List:
        """TODO"""
        return [self.pulse, self.channel]

    @property
    def channel(self) -> PulseChannel:
        """TODO"""
        return self._channel

    @property
    def pulse(self) -> Pulse:
        """TODO"""
        return self._pulse

    @property
    def pulse_name(self) -> str:
        return self.pulse.name

    def __repr__(self):
        """TODO"""
        return "{}({}, {}, name={})".format(self.__class__.__name__, self.pulse, self.channel,
                                            self.name)
