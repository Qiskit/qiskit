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
Timeslots for channels.
"""
from collections import defaultdict
import itertools
from typing import Tuple, Union, Optional

from .channels import Channel
from .exceptions import PulseError


# pylint: disable=missing-return-doc


class Interval:
    """Time interval."""

    def __init__(self, start: int, stop: int):
        """Create an interval, (start, stop).

        Args:
            start: Starting value of interval
            stop: Stopping value of interval

        Raises:
            PulseError: when invalid time or duration is specified
        """
        if start < 0:
            raise PulseError("Cannot create Interval with negative starting value")
        if stop < 0:
            raise PulseError("Cannot create Interval with negative stopping value")
        if start > stop:
            raise PulseError("Cannot create Interval with value start after stop")
        self._start = start
        self._stop = stop

    @property
    def start(self):
        """Start of interval."""
        return self._start

    @property
    def stop(self):
        """Stop of interval."""
        return self._stop

    @property
    def duration(self):
        """Duration of this interval."""
        return self.stop - self.start

    def has_overlap(self, interval: 'Interval') -> bool:
        """Check if self has overlap with `interval`.

        Args:
            interval: interval to be examined

        Returns:
            bool: True if self has overlap with `interval` otherwise False
        """
        return self.start < interval.stop and interval.start < self.stop

    def shift(self, time: int) -> 'Interval':
        """Return a new interval shifted by `time` from self

        Args:
            time: time to be shifted

        Returns:
            Interval: interval shifted by `time`
        """
        return Interval(self.start + time, self.stop + time)

    def __eq__(self, other):
        """Two intervals are the same if they have the same starting and stopping values.

        Args:
            other (Interval): other Interval

        Returns:
            bool: are self and other equal.
        """
        return self.start == other.start and self.stop == other.stop

    def stops_before(self, other):
        """Whether intervals stops at value less than or equal to the
        other interval's starting time.

        Args:
            other (Interval): other Interval

        Returns:
            bool: are self and other equal.
        """
        return self.stop <= other.start

    def starts_after(self, other):
        """Whether intervals starts at value greater than or equal to the
        other interval's stopping time.

        Args:
            other (Interval): other Interval

        Returns:
            bool: are self and other equal.
        """
        return self.start >= other.stop

    def __repr__(self):
        """Return a readable representation of Interval Object"""
        return "{}({}, {})".format(self.__class__.__name__, self.start, self.stop)


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

    @property
    def start(self):
        """Start of timeslot."""
        return self.interval.start

    @property
    def stop(self):
        """Stop of timeslot."""
        return self.interval.stop

    @property
    def duration(self):
        """Duration of this timeslot."""
        return self.interval.duration

    def shift(self, time: int) -> 'Timeslot':
        """Return a new Timeslot shifted by `time`.

        Args:
            time: time to be shifted
        """
        return Timeslot(self.interval.shift(time), self.channel)

    def has_overlap(self, other: 'Timeslot') -> bool:
        """Check if self has overlap with `interval`.

        Args:
            other: Other Timeslot to check for overlap with

        Returns:
            bool: True if intervals overlap and are on the same channel
        """
        return self.interval.has_overlap(other) and self.channel == other.channel

    def __eq__(self, other) -> bool:
        """Two time-slots are the same if they have the same interval and channel.

        Args:
            other (Timeslot): other Timeslot
        """
        return self.interval == other.interval and self.channel == other.channel

    def __repr__(self):
        """Return a readable representation of Timeslot Object"""
        return "{}({}, {})".format(self.__class__.__name__,
                                   self.channel,
                                   (self.interval.start, self.interval.stop))


class TimeslotCollection:
    """Collection of `Timeslot`s."""

    def __init__(self, *timeslots: Union[Timeslot, 'TimeslotCollection']):
        """Create a new time-slot collection.

        Args:
            *timeslots: list of time slots
        Raises:
            PulseError: when overlapped time slots are specified
        """
        self._table = defaultdict(list)

        for timeslot in timeslots:
            if isinstance(timeslot, TimeslotCollection):
                self._merge_timeslot_collection(timeslot)
            else:
                self._merge_timeslot(timeslot)

    @property
    def timeslots(self) -> Tuple[Timeslot]:
        """Sorted tuple of `Timeslot`s in collection."""
        return tuple(itertools.chain.from_iterable(self._table.values()))

    @property
    def channels(self) -> Tuple[Timeslot]:
        """Channels within the timeslot collection."""
        return tuple(k for k, v in self._table.items() if v)

    @property
    def start_time(self) -> int:
        """Return earliest start time in this collection."""
        return self.ch_start_time(*self.channels)

    @property
    def stop_time(self) -> int:
        """Return maximum time of timeslots over all channels."""
        return self.ch_stop_time(*self.channels)

    @property
    def duration(self) -> int:
        """Return maximum duration of timeslots over all channels."""
        return self.stop_time

    def _merge_timeslot_collection(self, other: 'TimeslotCollection'):
        """Mutably merge timeslot collections into this TimeslotCollection.

        Args:
            other: TimeSlotCollection to merge
        """
        for channel, other_ch_timeslots in other._table.items():
            if channel not in self._table:
                self._table[channel] += other_ch_timeslots  # extend to copy items
            else:
                # if channel is in self there might be an overlap
                for idx, other_ch_timeslot in enumerate(other_ch_timeslots):
                    insert_idx = self._merge_timeslot(other_ch_timeslot)
                    if insert_idx == len(self._table[channel]) - 1:
                        # Timeslot was inserted at end of list. The rest can be appended.
                        self._table[channel] += other_ch_timeslots[idx + 1:]
                        break

    def _merge_timeslot(self, timeslot: Timeslot) -> int:
        """Mutably merge timeslots into this TimeslotCollection.

        Note timeslots are sorted internally on their respective channel

        Args:
            timeslot: Timeslot to merge

        Returns:
            int: Return the index in which timeslot was inserted

        Raises:
            PulseError: If timeslots overlap
        """
        interval = timeslot.interval
        ch_timeslots = self._table[timeslot.channel]

        insert_idx = len(ch_timeslots)

        # merge timeslots by insertion sort.
        # Worst case O(n_channels), O(1) for append
        # could be improved by implementing an interval tree
        for ch_timeslot in reversed(ch_timeslots):
            ch_interval = ch_timeslot.interval

            if interval.start >= ch_interval.stop:
                break
            if interval.has_overlap(ch_interval):
                overlap_start = interval.start if interval.start > ch_interval.start \
                    else ch_interval.start
                overlap_end = ch_interval.stop if interval.stop > ch_interval.stop \
                    else interval.stop
                raise PulseError("Overlap on channel {0} over time range [{1}, {2}]"
                                 "".format(timeslot.channel, overlap_start, overlap_end))

            insert_idx -= 1

        ch_timeslots.insert(insert_idx, timeslot)

        return insert_idx

    def ch_timeslots(self, channel: Channel) -> Tuple[Timeslot]:
        """Sorted tuple of `Timeslot`s for channel in this TimeslotCollection."""
        if channel in self._table:
            return tuple(self._table[channel])

        return tuple()

    def ch_start_time(self, *channels: Channel) -> int:
        """Return earliest start time in this collection.

        Args:
            *channels: Channels over which to obtain start_time.
        """
        timeslots = list(itertools.chain(*(self._table[chan] for chan in channels
                                           if chan in self._table)))
        if timeslots:
            return min(timeslot.start for timeslot in timeslots)

        return 0

    def ch_stop_time(self, *channels: Channel) -> int:
        """Return maximum time of timeslots over all channels.

        Args:
            *channels: Channels over which to obtain stop time.
        """
        timeslots = list(itertools.chain(*(self._table[chan] for chan in channels
                                           if chan in self._table)))
        if timeslots:
            return max(timeslot.stop for timeslot in timeslots)

        return 0

    def ch_duration(self, *channels: Channel) -> int:
        """Return maximum duration of timeslots over all channels.

        Args:
            *channels: Channels over which to obtain the duration.
        """
        return self.ch_stop_time(*channels)

    def is_mergeable_with(self, other: 'TimeslotCollection') -> bool:
        """Return if self is mergeable with `timeslots`.

        Args:
            other: TimeslotCollection to be checked for mergeability
        """
        common_channels = set(self.channels) & set(other.channels)

        for channel in common_channels:
            ch_timeslots = self.ch_timeslots(channel)
            other_ch_timeslots = other.ch_timeslots(channel)
            if ch_timeslots[-1].stop < other_ch_timeslots[0].start:
                continue  # We are appending along this channel

            i = 0  # iterate through this
            j = 0  # iterate through other
            while i < len(ch_timeslots) and j < len(other_ch_timeslots):
                if ch_timeslots[i].interval.has_overlap(other_ch_timeslots[j].interval):
                    return False
                if ch_timeslots[i].stop <= other_ch_timeslots[j].start:
                    i += 1
                else:
                    j += 1
        return True

    def merge(self, timeslots: 'TimeslotCollection') -> 'TimeslotCollection':
        """Return a new TimeslotCollection with `timeslots` merged into it.

        Args:
            timeslots: TimeslotCollection to be merged
        """
        return TimeslotCollection(self, timeslots)

    def shift(self, time: int) -> 'TimeslotCollection':
        """Return a new TimeslotCollection shifted by `time`.

        Args:
            time: time to be shifted by
        """
        new_tc = TimeslotCollection()
        new_tc_table = new_tc._table
        for channel, timeslots in self._table.items():
            new_tc_table[channel] = [tc.shift(time) for tc in timeslots]
        return new_tc

    def complement(self, stop_time: Optional[int] = None) -> 'TimeslotCollection':
        """Return a complement TimeSlotCollection containing all unoccupied Timeslots
        within this TimeSlotCollection.


        Args:
            stop_time: Final time too which complement Timeslot's will be returned.
                If not set, defaults to last time in this TimeSlotCollection
        """

        timeslots = []

        stop_time = stop_time or self.stop_time

        for channel in self.channels:
            curr_time = 0
            for timeslot in self.ch_timeslots(channel):
                next_time = timeslot.interval.start
                if next_time-curr_time > 0:
                    timeslots.append(Timeslot(Interval(curr_time, next_time), channel))

                curr_time = timeslot.interval.stop

            # pad out channel to stop_time
            if stop_time-curr_time > 0:
                timeslots.append(Timeslot(Interval(curr_time, stop_time), channel))

        return TimeslotCollection(*timeslots)

    def __eq__(self, other) -> bool:
        """Two time-slot collections are the same if they have the same time-slots.

        Args:
            other (TimeslotCollection): other TimeslotCollection
        """
        if set(self.channels) != set(other.channels):
            return False

        for channel in self.channels:
            if self.ch_timeslots(channel) != self.ch_timeslots(channel):
                return False
        return True

    def __repr__(self):
        """Return a readable representation of TimeslotCollection Object"""
        rep = dict()
        for key, val in self._table.items():
            rep[key] = [(timeslot.start, timeslot.stop) for timeslot in val]
        return self.__class__.__name__ + str(rep)
