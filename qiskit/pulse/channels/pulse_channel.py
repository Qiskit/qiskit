# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Channels.
"""
from abc import ABCMeta, abstractmethod


class Channel(metaclass=ABCMeta):
    """Pulse channel."""

    prefix = None

    @abstractmethod
    def __init__(self, index: int = None):
        self._index = index

    @property
    def index(self) -> int:
        """Return the index of this channel."""
        return self._index

    @property
    def name(self) -> str:
        """Return the name of this channel."""
        return '%s%d' % (self.__class__.prefix, self._index)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self._index)

    def __eq__(self, other):
        """Two channels are the same if they are of the same type, and have the same index.

        Args:
            other (Channel): other PulseChannel

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self._index == other._index:
            return True
        return False

    def __hash__(self):
        return hash((type(self), self._index))


class AcquireChannel(Channel):
    """Acquire channel."""

    prefix = 'a'

    def __init__(self, index):
        """Create new acquire channel.

        Args:
            index (int): Index of the channel.
        """
        super().__init__(index)


class SnapshotChannel(Channel):
    """Snapshot channel."""

    prefix = 's'

    def __init__(self):
        """Create new snapshot channel."""
        super().__init__(0)


class MemorySlot(Channel):
    """Memory slot."""

    prefix = 'm'

    def __init__(self, index):
        """Create new memory slot.

        Args:
            index (int): Index of the channel.
        """
        super().__init__(index)


class RegisterSlot(Channel):
    """Classical resister slot channel."""

    prefix = 'c'

    def __init__(self, index):
        """Create new register slot.

        Args:
            index (int): Index of the channel.
        """
        super().__init__(index)
