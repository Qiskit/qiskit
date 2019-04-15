# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Channels support signal output.
"""
from abc import abstractmethod
from typing import Tuple

from qiskit.pulse.exceptions import PulseError
from .pulse_channel import Channel


class LoRange:
    """Range of LO frequency."""

    def __init__(self, lower_bound: float, upper_bound: float):
        self._lb = lower_bound
        self._ub = upper_bound

    def includes(self, lo_freq: float) -> bool:
        """Return if this range includes `lo_freq` or not.

        Args:
            lo_freq: LO frequency to be checked

        Returns:
            bool: True if lo_freq is included in this range, otherwise False
        """
        if self._lb <= lo_freq <= self._ub:
            return True
        return False

    @property
    def lower_bound(self) -> float:
        """Lower bound of this LO range"""
        return self._lb

    @property
    def upper_bound(self) -> float:
        """Upper bound of this LO range"""
        return self._ub

    def __repr__(self):
        return "%s(%f, %f)" % (self.__class__.__name__, self._lb, self._ub)


class OutputChannel(Channel):
    """Output Channel."""

    @abstractmethod
    def __init__(self,
                 index: int,
                 lo_freq: float = None,
                 lo_freq_range: Tuple[float, float] = (0, float("inf"))):
        super().__init__(index)
        self._lo_freq = lo_freq

        if (not isinstance(lo_freq_range, tuple)) or len(lo_freq_range) != 2:
            raise PulseError("Invalid form of lo_freq_range is specified.")
        self._lo_freq_range = LoRange(*lo_freq_range)

        if self._lo_freq:
            if not self._lo_freq_range.includes(self._lo_freq):
                raise PulseError("lo_freq %f must be within lo_freq_range %s" %
                                 (self._lo_freq, self._lo_freq_range))

    @property
    def lo_freq(self) -> float:
        """Get the default lo frequency."""
        return self._lo_freq

    @property
    def lo_freq_range(self) -> LoRange:
        """Get the feasible range of LO frequency."""
        return self._lo_freq_range

    def __eq__(self, other):
        """Two output channels are the same if they are of the same type, and
        have the same index and lo_freq.

        Args:
            other (OutputChannel): other OutputChannel

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self._index == other._index and \
                self._lo_freq == other._lo_freq:
            return True
        return False

    def __hash__(self):
        return hash((super().__hash__(), self._lo_freq))


class DriveChannel(OutputChannel):
    """Drive Channel."""

    prefix = 'd'

    def __init__(self, index: int,
                 lo_freq: float = None,
                 lo_freq_range: Tuple[float, float] = (0, float("inf"))):
        """Create new drive (d) channel.

        Args:
            index (int): index of the channel
            lo_freq (float): default frequency of LO (local oscillator)
            lo_freq_range (tuple): feasible range of LO frequency
        """
        super().__init__(index, lo_freq, lo_freq_range)


class ControlChannel(OutputChannel):
    """Control Channel."""

    prefix = 'u'

    def __init__(self, index: int,
                 lo_freq: float = None,
                 lo_freq_range: Tuple[float, float] = (0, float("inf"))):
        """Create new control (u) channel.

        Args:
            index (int): index of the channel
            lo_freq (float): default frequency of LO (local oscillator)
            lo_freq_range (tuple): feasible range of LO frequency
        """
        super().__init__(index, lo_freq, lo_freq_range)


class MeasureChannel(OutputChannel):
    """Measure Channel."""

    prefix = 'm'

    def __init__(self, index: int,
                 lo_freq: float = None,
                 lo_freq_range: Tuple[float, float] = (0, float("inf"))):
        """Create new measurement (m) channel.

        Args:
            index (int): index of the channel
            lo_freq (float): default frequency of LO (local oscillator)
            lo_freq_range (tuple): feasible range of LO frequency
        """
        super().__init__(index, lo_freq, lo_freq_range)
