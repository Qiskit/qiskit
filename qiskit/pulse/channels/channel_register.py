# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
PulseChannel register.
"""
import logging
from typing import Union, List

from qiskit.circuit import Register
from qiskit.exceptions import QiskitError, QiskitIndexError
from qiskit.pulse.exceptions import ChannelsError
from .backend_channels import OutputChannel
from .pulse_channel import PulseChannel

logger = logging.getLogger(__name__)


class ChannelRegister(Register):
    """Implement a channel register."""

    def __init__(self, channel_cls, size, name=None):
        """Create a new channel register.
        """
        super().__init__(size, name)

        if not issubclass(channel_cls, PulseChannel):
            raise ChannelsError("Unknown channel class: %s", channel_cls.__name__)

        self.channel_cls = channel_cls
        self._channels = [self.channel_cls(ind) for ind in range(self.size)]

    def __repr__(self):
        """Return the official string representing the register."""
        return "%s(%s, %d, '%s')" % (self.__class__.__qualname__,
                                     self.channel_cls.__name__,
                                     self.size, self.name)

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
            raise QiskitError("expected integer or slice index into register")
        self.check_range(key)   # TODO: need more checks!
        if isinstance(key, list):  # list of channel indices
            if max(key) < len(self):    # TODO: need to move check_range()
                return [self._channels[i] for i in key]
            else:
                raise QiskitIndexError('register index out of range')
        else:
            return self._channels[key]

    def __iter__(self):
        """
        Returns:
            iterator: an iterator over the channels of the register.
        """
        yield from self._channels

    def __eq__(self, other):
        """Two channel registers are the same if they are of the same type
         have the same channel class, name and size.

        Args:
            other (ChannelRegister): other ChannelRegister

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self.channel_cls == other.channel_cls and \
                self.name == other.name and \
                self.size == other.size:
            return True
        return False

    def __hash__(self):
        """Make object hashable."""
        return hash((type(self), self.channel_cls, self.name, self.size))


class OutputChannelRegister(ChannelRegister):
    """Implement an output channel register."""

    def __init__(self, channel_cls, size: int, name: str = None, lo_freqs: List[float] = None):
        """Create a new output channel register.
        """
        if not issubclass(channel_cls, OutputChannel):
            raise ChannelsError("Unknown output channel class: %s", channel_cls.__name__)

        super().__init__(channel_cls, size, name)

        if lo_freqs:
            self.lo_freqs = lo_freqs
        else:
            self.lo_freqs = [None] * size

    def __eq__(self, other):
        """Two channel registers are the same if they are of the same type
         have the same channel class, name and size.

        Args:
            other (OutputChannelRegister): other OutputChannelRegister

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self.channel_cls == other.channel_cls and \
                self.name == other.name and \
                self.size == other.size and \
                self.lo_freqs == other.lo_freqs:
            return True
        return False

    def __hash__(self):
        """Make object hashable."""
        return hash((type(self), self.channel_cls, self.name, self.size, self.lo_freqs))
