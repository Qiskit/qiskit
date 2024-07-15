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
from __future__ import annotations
import numpy as np

from .channels import DriveChannel, MeasureChannel
from .exceptions import PulseError


def _assert_nested_dict_equal(a: dict, b: dict):
    if len(a) != len(b):
        return False
    for key in a:
        if key in b:
            if isinstance(a[key], dict):
                if not _assert_nested_dict_equal(a[key], b[key]):
                    return False
            elif isinstance(a[key], np.ndarray):
                if not np.all(a[key] == b[key]):
                    return False
            else:
                if a[key] != b[key]:
                    return False
        else:
            return False
    return True


class Kernel:
    """Settings for this Kernel, which is responsible for integrating time series (raw) data
    into IQ points.
    """

    def __init__(self, name: str | None = None, **params):
        """Create new kernel.

        Args:
            name: Name of kernel to be used
            params: Any settings for kerneling.
        """
        self.name = name
        self.params = params

    def __repr__(self):
        name_repr = "'" + self.name + "', "
        params_repr = ", ".join(f"{str(k)}={str(v)}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({name_repr}{params_repr})"

    def __eq__(self, other):
        if isinstance(other, Kernel):
            return _assert_nested_dict_equal(self.__dict__, other.__dict__)
        return False


class Discriminator:
    """Setting for this Discriminator, which is responsible for classifying kerneled IQ points
    into 0/1 state results.
    """

    def __init__(self, name: str | None = None, **params):
        """Create new discriminator.

        Args:
            name: Name of discriminator to be used
            params: Any settings for discrimination.
        """
        self.name = name
        self.params = params

    def __repr__(self):
        name_repr = "'" + self.name + "', " or ""
        params_repr = ", ".join(f"{str(k)}={str(v)}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({name_repr}{params_repr})"

    def __eq__(self, other):
        if isinstance(other, Discriminator):
            return _assert_nested_dict_equal(self.__dict__, other.__dict__)
        return False


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
        return f"{self.__class__.__name__}({self._lb:f}, {self._ub:f})"

    def __eq__(self, other):
        """Two LO ranges are the same if they are of the same type, and
        have the same frequency range

        Args:
            other (LoRange): other LoRange

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and self._ub == other._ub and self._lb == other._lb:
            return True
        return False


class LoConfig:
    """Pulse channel LO frequency container."""

    def __init__(
        self,
        channel_los: dict[DriveChannel | MeasureChannel, float] | None = None,
        lo_ranges: dict[DriveChannel | MeasureChannel, LoRange | tuple[int, int]] | None = None,
    ):
        """Lo channel configuration data structure.

        Args:
            channel_los: Dictionary of mappings from configurable channel to lo
            lo_ranges: Dictionary of mappings to be enforced from configurable channel to `LoRange`

        Raises:
            PulseError: If channel is not configurable or set lo is out of range.

        """
        self._q_lo_freq: dict[DriveChannel, float] = {}
        self._m_lo_freq: dict[MeasureChannel, float] = {}
        self._lo_ranges: dict[DriveChannel | MeasureChannel, LoRange] = {}

        lo_ranges = lo_ranges if lo_ranges else {}
        for channel, freq in lo_ranges.items():
            self.add_lo_range(channel, freq)

        channel_los = channel_los if channel_los else {}
        for channel, freq in channel_los.items():
            self.add_lo(channel, freq)

    def add_lo(self, channel: DriveChannel | MeasureChannel, freq: float):
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
            raise PulseError(f"Specified channel {channel.name} cannot be configured.")

    def add_lo_range(
        self, channel: DriveChannel | MeasureChannel, lo_range: LoRange | tuple[int, int]
    ):
        """Add lo range to configuration.

        Args:
            channel: Channel to add lo range for
            lo_range: Lo range to add

        """
        if isinstance(lo_range, (list, tuple)):
            lo_range = LoRange(*lo_range)
        self._lo_ranges[channel] = lo_range

    def check_lo(self, channel: DriveChannel | MeasureChannel, freq: float) -> bool:
        """Check that lo is valid for channel.

        Args:
            channel: Channel to validate lo for
            freq: lo frequency
        Raises:
            PulseError: If freq is outside of channels range
        Returns:
            True if lo is valid for channel
        """
        lo_ranges = self._lo_ranges
        if channel in lo_ranges:
            lo_range = lo_ranges[channel]
            if not lo_range.includes(freq):
                raise PulseError(f"Specified LO freq {freq:f} is out of range {lo_range}")
        return True

    def channel_lo(self, channel: DriveChannel | MeasureChannel) -> float:
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

        raise PulseError(f"Channel {channel} is not configured")

    @property
    def qubit_los(self) -> dict[DriveChannel, float]:
        """Returns dictionary mapping qubit channels (DriveChannel) to los."""
        return self._q_lo_freq

    @property
    def meas_los(self) -> dict[MeasureChannel, float]:
        """Returns dictionary mapping measure channels (MeasureChannel) to los."""
        return self._m_lo_freq
