# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Configurations for pulse experiments.
"""
from typing import Dict

from .channels import OutputChannel, DriveChannel, MeasureChannel
from .exceptions import PulseError


class LoConfig:
    """Dictionary of user LO frequency by channel"""

    def __init__(self, user_lo_dic: Dict[OutputChannel, float] = None):
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

    def qubit_lo_dict(self):
        """Return items of qubit LOs."""
        return self._q_lo_freq.items()

    def meas_lo_dict(self):
        """Return items of meas LOs."""
        return self._m_lo_freq.items()
