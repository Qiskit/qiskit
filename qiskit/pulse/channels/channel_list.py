# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
PulseChannel list.
"""
import logging
from typing import Union, List

from qiskit.exceptions import QiskitError, QiskitIndexError
from qiskit.pulse.exceptions import ChannelsError
from .pulse_channel import PulseChannel, AcquireChannel, SnapshotChannel

logger = logging.getLogger(__name__)


class ChannelList:
    """Implement a channel list."""

    def __init__(self, channel_cls, size):
        """Create a new channel list.
        """
        self._size = size

        if not issubclass(channel_cls, PulseChannel):
            raise ChannelsError("Unknown channel class: %s", channel_cls.__name__)

        self._channel_cls = channel_cls
        self._channels = [self._channel_cls(i) for i in range(self._size)]

    def __repr__(self):
        """Return the official string representing the list."""
        return "%s(%s, %d)" % (self.__class__.__qualname__,
                               self._channel_cls.__name__,
                               self._size)

    def __getitem__(self, key) -> Union[PulseChannel, List[PulseChannel]]:
        """
        Arg:
            key (int|slice|list): index of the channel to be retrieved.

        Returns:
            PulseChannel|list(PulseChannel): a channel if key is int.
                If key is a slice, return a `list(PulseChannel)`.

        Raises:
            QiskitError: if the `key` is not an integer.
            QiskitIndexError: if the `key` is not in the range
                `(0, self.size)`.
        """
        if not isinstance(key, (int, slice, list)):
            raise QiskitError("expected integer or slice index into list")
        try:
            if isinstance(key, list):  # list of channel indices
                return [self._channels[i] for i in key]
            else:
                return self._channels[key]
        except IndexError:
            raise QiskitIndexError('channel list index out of range')

    def __iter__(self):
        """
        Returns:
            iterator: an iterator over the channels of the list.
        """
        yield from self._channels

    def __len__(self):
        """
        Returns:
            int: length of this channel list.
        """
        return self._size

    def __eq__(self, other):
        """Two channel lists are the same if they are of the same type
         have the same size and channel class.

        Args:
            other (ChannelList): other ChannelList

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self._size == other._size and \
                self._channel_cls == other._channel_cls:
            return True
        return False

    def __hash__(self):
        """Make object hashable."""
        return hash((super().__hash__(), self._channel_cls))


class AcquireChannelList(ChannelList):
    """Acquire channel list."""

    def __init__(self, size: int):
        """Create a new acquire channel list.
        """
        super().__init__(AcquireChannel, size)


class SnapshotChannelList(ChannelList):
    """Snapshot channel list."""

    def __init__(self, size: int):
        """Create a new snapshot channel list.
        """
        super().__init__(SnapshotChannel, size)
