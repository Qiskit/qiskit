# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Configurations for pulse experiments.
"""
from typing import Dict

from qiskit.pulse.channels import OutputChannel
from qiskit.pulse.exceptions import PulseError


class UserLoDict:
    """Dictionary of user LO frequency by channel"""

    def __init__(self, user_lo_dic: Dict[OutputChannel, float] = None):
        self._user_lo_dic = {}
        if user_lo_dic:
            for channel, user_lo in user_lo_dic.items():
                if not channel.lo_freq_range.includes(user_lo):
                    raise PulseError("Specified LO freq %f is out of range %s" %
                                     (user_lo, channel.lo_freq_range))
                self._user_lo_dic[channel] = user_lo

    # TODO: what should we publish? (with keeping this object immutable)
    def items(self):
        """Return items of this user LO dictionary"""
        return self._user_lo_dic.items()
