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
from typing import Dict, Union, Tuple, Optional

from .channels import PulseChannel, DriveChannel, MeasureChannel
from .exceptions import PulseError


class LoRange:
    """Range of LO frequency."""

    def __init__(self, lower_bound: float, upper_bound: float):
        self._lb = lower_bound
        self._ub = upper_bound

    def includes(self, lo_freq: complex) -> bool:
        """Whether `lo_freq` is within the `LoRange`.

        Args:
            lo_freq: LO frequency to be validated

        Returns:
            bool: True if lo_freq is included in this range, otherwise False
        """
        if self._lb <= abs(lo_freq) <= self._ub:
            return True
        return False

    @property
    def lower_bound(self) -> float:
        """Lower bound of the LO range"""
        return self._lb

    @property
    def upper_bound(self) -> float:
        """Upper bound of the LO range"""
        return self._ub

    def __repr__(self):
        return "%s(%f, %f)" % (self.__class__.__name__, self._lb, self._ub)

    def __eq__(self, other):
        """Two LO ranges are the same if they are of the same type, and
        have the same frequency range

        Args:
            other (LoRange): other LoRange

        Returns:
            bool: are self and other equal.
        """
        if (type(self) is type(other) and
                self._ub == other._ub and
                self._lb == other._lb):
            return True
        return False


class LoConfig:
    """Pulse channel LO frequency container."""

    def __init__(self, channel_los: Optional[Dict[PulseChannel, float]] = None,
                 lo_ranges: Optional[Dict[PulseChannel, Union[LoRange, Tuple[int]]]] = None):
        """Lo channel configuration data structure.

        Args:
            channel_los: Dictionary of mappings from configurable channel to lo
            lo_ranges: Dictionary of mappings to be enforced from configurable channel to `LoRange`

        Raises:
            PulseError: If channel is not configurable or set lo is out of range.

        """
        self._q_lo_freq = {}
        self._m_lo_freq = {}
        self._lo_ranges = {}

        lo_ranges = lo_ranges if lo_ranges else {}
        for channel, freq in lo_ranges.items():
            self.add_lo_range(channel, freq)

        channel_los = channel_los if channel_los else {}
        for channel, freq in channel_los.items():
            self.add_lo(channel, freq)

    def add_lo(self, channel: Union[DriveChannel, MeasureChannel], freq: float):
        """Add a lo mapping for a channel."""
        if isinstance(channel, DriveChannel):
            # add qubit_lo_freq
            self.check_lo(channel, freq)
            self._q_lo_freq[channel] = freq
        elif isinstance(channel, MeasureChannel):
            # add meas_lo_freq
            self.check_lo(channel, freq)
            self._m_lo_freq[channel] = freq
        else:
            raise PulseError("Specified channel %s cannot be configured." %
                             channel.name)

    def add_lo_range(self, channel: DriveChannel, lo_range: Union[LoRange, Tuple[int]]):
        """Add lo range to configuration.

        Args:
            channel: Channel to add lo range for
            lo_range: Lo range to add

        """
        if isinstance(lo_range, (list, tuple)):
            lo_range = LoRange(*lo_range)
        self._lo_ranges[channel] = lo_range

    def check_lo(self, channel: Union[DriveChannel, MeasureChannel], freq: float) -> bool:
        """Check that lo is valid for channel.

        Args:
            channel: Channel to validate lo for
            freq: lo frequency
        Raises:
            PulseError: If freq is outside of channels range
        """
        lo_ranges = self._lo_ranges
        if channel in lo_ranges:
            lo_range = lo_ranges[channel]
            if not lo_range.includes(freq):
                raise PulseError("Specified LO freq %f is out of range %s" %
                                 (freq, lo_range))

    def channel_lo(self, channel: Union[DriveChannel, MeasureChannel]) -> float:
        """Return channel lo.

        Args:
            channel: Channel to get lo for
        Raises:
            PulseError: If channel is not configured
        Returns:
            Lo of supplied channel if present
        """
        if isinstance(channel, DriveChannel):
            if channel in self.qubit_los:
                return self.qubit_los[channel]

        if isinstance(channel, MeasureChannel):
            if channel in self.meas_los:
                return self.meas_los[channel]

        raise PulseError('Channel %s is not configured' % channel)

    @property
    def qubit_los(self) -> Dict:
        """Returns dictionary mapping qubit channels (DriveChannel) to los."""
        return self._q_lo_freq

    @property
    def meas_los(self) -> Dict:
        """Returns dictionary mapping measure channels (MeasureChannel) to los."""
        return self._m_lo_freq
