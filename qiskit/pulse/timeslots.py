# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Timeslots for channels.
"""
from collections import defaultdict
import itertools
import logging
from typing import List, Tuple

from .channels import Channel
from .exceptions import PulseError

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

    def shift(self, time: int) -> 'Interval':
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

    def shift(self, time: int) -> 'Timeslot':
        """Return a new Timeslot shifted by `time`.

        Args:
            time: time to be shifted

        Returns:
            A new Timeslot object shifted by `time`.
        """
        return Timeslot(self.interval.shift(time), self.channel)

    def __eq__(self, other):
        """Two time-slots are the same if they have the same interval and channel.

        Args:
            other (Timeslot): other Timeslot

        Returns:
            bool: are self and other equal.
        """
        if self.interval == other.interval and self.channel == other.channel:
            return True
        return False


class TimeslotCollection:
    """Collection of `Timeslot`s."""

    def __init__(self, *timeslots: List[Timeslot]):
        """Create a new time-slot collection.

        Args:
            timeslots: list of time slots
        Raises:
            PulseError: when overlapped time slots are specified
        """
        self._table = defaultdict(list)

        for slot in timeslots:
            for interval in self._table[slot.channel]:
                if slot.interval.has_overlap(interval):
                    raise PulseError("Cannot create TimeslotCollection from overlapped timeslots")
            self._table[slot.channel].append(slot.interval)

        self._timeslots = tuple(timeslots)

    @property
    def timeslots(self) -> Tuple[Timeslot]:
        """`Timeslot`s in collection."""
        return self._timeslots

    @property
    def channels(self):
        """Channels within the timeslot collection.

        Returns:
            tuple
        """
        return tuple(self._table.keys())

    @property
    def start_time(self) -> int:
        """Return earliest start time in this collection.

        Returns:
            The earliest start time in this collection.
        """
        return self.ch_start_time(*self.channels)

    @property
    def stop_time(self) -> int:
        """Return maximum time of timeslots over all channels.

        Returns:
            The latest stop time in this collection.
        """
        return self.ch_stop_time(*self.channels)

    @property
    def duration(self) -> int:
        """Return maximum duration of timeslots over all channels.

        Returns:
            The duration over all chanels.
        """
        return self.stop_time - 0

    def ch_start_time(self, *channels: List[Channel]) -> int:
        """Return earliest start time in this collection.

        Args:
            channels: Channels over which to obtain start_time.

        Returns:
            The earliest start time over all channels.
        """
        intervals = list(itertools.chain(*(self._table[chan] for chan in channels
                                           if chan in self._table)))
        if intervals:
            return min((interval.begin for interval in intervals))
        return 0

    def ch_stop_time(self, *channels: List[Channel]) -> int:
        """Return maximum time of timeslots over all channels.

        Args:
            default(int, optional): default value used when this collection is empty
            channel (Channel): Optional channel

        Returns:
            The latest stop time over all channels.
        """
        intervals = list(itertools.chain(*(self._table[chan] for chan in channels
                                           if chan in self._table)))
        if intervals:
            return max((interval.end for interval in intervals))
        return 0

    def ch_duration(self, *channels: List[Channel]) -> int:
        """Return maximum duration of timeslots over all channels.

        Args:
            channels: Channels over which to obtain start_time.
        Returns:
            The maximum duration over all channels.
        """
        return self.ch_stop_time(*channels) - 0

    def is_mergeable_with(self, timeslots: 'TimeslotCollection') -> bool:
        """Return if self is mergeable with `timeslots`.

        Args:
            timeslots: TimeslotCollection to be checked

        Returns:
            True if self is mergeable with `timeslots`, otherwise False.
        """
        for slot in timeslots.timeslots:
            for interval in self._table[slot.channel]:
                if slot.interval.has_overlap(interval):
                    return False
        return True

    def merged(self, timeslots: 'TimeslotCollection') -> 'TimeslotCollection':
        """Return a new TimeslotCollection merged with a specified `timeslots`

        Args:
            timeslots: TimeslotCollection to be merged

        Returns:
            A new TimeslotCollection object merged with a specified `timeslots`.
        """
        slots = [Timeslot(slot.interval, slot.channel) for slot in self.timeslots]
        slots.extend([Timeslot(slot.interval, slot.channel) for slot in timeslots.timeslots])
        return TimeslotCollection(*slots)

    def shift(self, time: int) -> 'TimeslotCollection':
        """Return a new TimeslotCollection shifted by `time`.

        Args:
            time: time to be shifted by
        """
        slots = [Timeslot(slot.interval.shift(time), slot.channel) for slot in self.timeslots]
        return TimeslotCollection(*slots)

    def __eq__(self, other):
        """Two time-slot collections are the same if they have the same time-slots.

        Args:
            other (TimeslotCollection): other TimeslotCollection

        Returns:
            bool: are self and other equal.
        """
        if self.timeslots == other.timeslots:
            return True
        return False
