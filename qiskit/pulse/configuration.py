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
Configurations for pulse experiments.
"""
from typing import Dict

from .channels import PulseChannel, DriveChannel, MeasureChannel
from .exceptions import PulseError


class LoConfig:
    """Pulse channel LO frequency container."""

    def __init__(self, user_lo_dic: Dict[PulseChannel, float] = None):
        self._q_lo_freq = {}
        self._m_lo_freq = {}

        if user_lo_dic:
            for channel, lo_freq in user_lo_dic.items():
                if isinstance(channel, DriveChannel):
                    # add qubit_lo_freq
                    if not channel.lo_freq_range.includes(lo_freq):
                        raise PulseError("Specified LO freq %f is out of range %s" %
                                         (lo_freq, channel.lo_freq_range))
                    self._q_lo_freq[channel] = lo_freq
                elif isinstance(channel, MeasureChannel):
                    # add meas_lo_freq
                    if not channel.lo_freq_range.includes(lo_freq):
                        raise PulseError("Specified LO freq %f is out of range %s" %
                                         (lo_freq, channel.lo_freq_range))
                    self._m_lo_freq[channel] = lo_freq
                else:
                    raise PulseError("Specified channel %s cannot be configured." %
                                     channel.name)

    def qubit_lo_dict(self) -> Dict:
        """Return items of qubit LOs.
        """
        return self._q_lo_freq

    def meas_lo_dict(self) -> Dict:
        """Return items of meas LOs."""
        return self._m_lo_freq
