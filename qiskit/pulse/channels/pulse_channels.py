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
from .channels import Channel


class LoRange:
    """Range of LO frequency."""

    def __init__(self, lower_bound: float, upper_bound: float):
        self._lb = lower_bound
        self._ub = upper_bound

    def includes(self, lo_freq: complex) -> bool:
        """Whether `lo_freq` is within the `LoRange`.

        Args:
            lo_freq: LO frequency to be checked

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


class PulseChannel(Channel):
    """Base class of Channel supporting pulse output."""

    @abstractmethod
    def __init__(self, index: int, lo_freq_range: Tuple[float, float] = (0, float("inf"))):
        super().__init__(index)

        if (not isinstance(lo_freq_range, tuple)) or len(lo_freq_range) != 2:
            raise PulseError("Invalid form of lo_freq_range is specified.")
        self._lo_freq_range = LoRange(*lo_freq_range)

    @property
    def lo_freq_range(self) -> LoRange:
        """Get the acceptable LO range."""
        return self._lo_freq_range


class DriveChannel(PulseChannel):
    """Drive Channel."""

    prefix = 'd'

    def __init__(self, index: int, lo_freq_range: Tuple[float, float] = (0, float("inf"))):
        """Create new drive (d) channel.

        Args:
            index: index of the channel
            lo_freq_range: feasible range of LO frequency
        """
        super().__init__(index, lo_freq_range)


class MeasureChannel(PulseChannel):
    """Measure Channel."""

    prefix = 'm'

    def __init__(self, index: int, lo_freq_range: Tuple[float, float] = (0, float("inf"))):
        """Create new measurement (m) channel.

        Args:
            index: index of the channel
            lo_freq_range: feasible range of LO frequency
        """
        super().__init__(index, lo_freq_range)


class ControlChannel(PulseChannel):
    """Control Channel."""

    prefix = 'u'

    def __init__(self, index: int):
        """Create new control (u) channel.

        Args:
            index: index of the channel
        """
        super().__init__(index, (0, float("inf")))
