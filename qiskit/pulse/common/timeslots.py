# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Timeslot occupancy for each channels.
"""
import logging
from collections import defaultdict
from typing import List, Optional, Tuple

from qiskit.pulse.channels import Channel
from qiskit.pulse.exceptions import PulseError

logger = logging.getLogger(__name__)


class Interval:
    """Time interval."""

    def __init__(self, begin: int, end: int):
        """Create an interval = (begin, end))

        Args:
            begin: begin time of this interval
            end: end time of this interval

        Raises:
            PulseError: when invalid time or duration is specified
        """
        if begin < 0:
            raise PulseError("Cannot create Interval with negative begin time")
        if end < 0:
            raise PulseError("Cannot create Interval with negative end time")
        self._begin = begin
        self._end = end

    @property
    def begin(self):
        """Begin time of this interval."""
        return self._begin

    @property
    def end(self):
        """End time of this interval."""
        return self._end

    @property
    def duration(self):
        """Duration of this interval."""
        return self._end - self._begin

    def has_overlap(self, interval: 'Interval') -> bool:
        """Check if self has overlap with `interval`.

        Args:
            interval: interval to be examined

        Returns:
            bool: True if self has overlap with `interval` otherwise False
        """
        if self.begin < interval.end and interval.begin < self.end:
            return True
        return False

    def shifted(self, time: int) -> 'Interval':
        """Return a new interval shifted by `time` from self

        Args:
            time: time to be shifted

        Returns:
            Interval: interval shifted by `time`
        """
        return Interval(self._begin + time, self._end + time)

    def __eq__(self, other):
        """Two intervals are the same if they have the same begin and end.

        Args:
            other (Interval): other Interval

        Returns:
            bool: are self and other equal.
        """
        if self._begin == other._begin and self._end == other._end:
            return True
        return False


class Timeslot:
    """Named tuple of (Interval, Channel)."""

    def __init__(self, interval: Interval, channel: Channel):
        self._interval = interval
        self._channel = channel

    @property
    def interval(self):
        """Interval of this time slot."""
        return self._interval

    @property
    def channel(self):
        """Channel of this time slot."""
        return self._channel

    def shifted(self, time: int) -> 'Timeslot':
        """Return a new Timeslot shifted by `time`.

        Args:
            time: time to be shifted

        Returns:
            A new Timeslot object shifted by `time`.
        """
        return Timeslot(self._interval.shifted(time), self._channel)

    def __eq__(self, other):
        """Two time-slots are the same if they have the same interval and channel.

        Args:
            other (Timeslot): other Timeslot

        Returns:
            bool: are self and other equal.
        """
        if self._interval == other._interval and self._channel == other._channel:
            return True
        return False


class TimeslotCollection:
    """Collection of `Timeslot`s."""

    def __init__(self, timeslots: List[Timeslot]):
        """Create a new time-slot collection.

        Args:
            timeslots: list of time slots
        Raises:
            PulseError: when overlapped time slots are specified
        """
        self._timeslots = tuple(timeslots)
        self._table = defaultdict(list)
        for slot in timeslots:
            for interval in self._table[slot.channel]:
                if slot.interval.has_overlap(interval):
                    raise PulseError("Cannot create TimeslotCollection from overlapped timeslots")
            self._table[slot.channel].append(slot.interval)

    @property
    def timeslots(self) -> Tuple[Timeslot, ...]:
        """Time slots of this occupancy."""
        return self._timeslots

    def is_mergeable_with(self, occupancy: 'TimeslotCollection') -> bool:
        """Return if self is mergeable with a specified `occupancy` or not.

        Args:
            occupancy: TimeslotCollection to be checked

        Returns:
            True if self is mergeable with `occupancy`, otherwise False.
        """
        for slot in occupancy._timeslots:
            for interval in self._table[slot.channel]:
                if slot.interval.has_overlap(interval):
                    return False
        return True

    def merged(self, occupancy: 'TimeslotCollection') -> Optional['TimeslotCollection']:
        """Return a new TimeslotCollection merged with a specified `occupancy`

        Args:
            occupancy: TimeslotCollection to be merged

        Returns:
            A new TimeslotCollection object merged with a specified `occupancy`.
        """
        slots = [Timeslot(slot.interval, slot.channel) for slot in self._timeslots]
        slots.extend([Timeslot(slot.interval, slot.channel) for slot in occupancy._timeslots])
        return TimeslotCollection(slots)

    def shifted(self, time: int) -> 'TimeslotCollection':
        """Return a new TimeslotCollection shifted by `time`.

        Args:
            time: time to be shifted

        Returns:
            A new TimeslotCollection object shifted by `time`.
        """
        slots = [Timeslot(slot.interval.shifted(time), slot.channel) for slot in self._timeslots]
        return TimeslotCollection(slots)

    def __eq__(self, other):
        """Two time-slot collections are the same if they have the same time-slots.

        Args:
            other (TimeslotCollection): other TimeslotCollection

        Returns:
            bool: are self and other equal.
        """
        if self._timeslots == other._timeslots:
            return True
        return False
