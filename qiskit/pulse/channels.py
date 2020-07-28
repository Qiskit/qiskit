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

"""This module defines Pulse Channels. Channels include:

  - transmit channels, which should subclass ``PulseChannel``
  - receive channels, such as ``AcquireChannel``
  - non-signal "channels" such as ``SnapshotChannel``, ``MemorySlot`` and ``RegisterChannel``.

Novel channel types can often utilize the ``ControlChannel``, but if this is not sufficient, new
channel types can be created. Then, they must be supported in the PulseQobj schema and the
assembler.
"""
from abc import ABCMeta

from qiskit.pulse.exceptions import PulseError


class Channel(metaclass=ABCMeta):
    """Base class of channels. Channels provide a Qiskit-side label for typical quantum control
    hardware signal channels. The final label -> physical channel mapping is the responsibility
    of the hardware backend. For instance, ``DriveChannel(0)`` holds instructions which the backend
    should map to the signal line driving gate operations on the qubit labeled (indexed) 0.
    """

    prefix = None  # type: str
    """A shorthand string prefix for characterizing the channel type."""

    def __init__(self, index: int):
        """Channel class.

        Args:
            index: Index of channel.

        Raises:
            PulseError: If ``index`` is not a nonnegative integer.
        """
        if not isinstance(index, int) or index < 0:
            raise PulseError('Channel index must be a nonnegative integer')
        self._index = index
        self._hash = None

    @property
    def index(self) -> int:
        """Return the index of this channel. The index is a label for a control signal line
        typically mapped trivially to a qubit index. For instance, ``DriveChannel(0)`` labels
        the signal line driving the qubit labeled with index 0.
        """
        return self._index

    @property
    def name(self) -> str:
        """Return the shorthand alias for this channel, which is based on its type and index."""
        return '%s%d' % (self.__class__.prefix, self._index)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self._index)

    def __eq__(self, other: 'Channel') -> bool:
        """Return True iff self and other are equal, specifically, iff they have the same type
        and the same index.

        Args:
            other: The channel to compare to this channel.

        Returns:
            True iff equal.
        """
        return type(self) is type(other) and self._index == other._index

    def __hash__(self):
        if self._hash is None:
            self._hash = hash((type(self), self._index))
        return self._hash


class PulseChannel(Channel, metaclass=ABCMeta):
    """Base class of transmit Channels. Pulses can be played on these channels."""
    pass


class DriveChannel(PulseChannel):
    """Drive channels transmit signals to qubits which enact gate operations."""
    prefix = 'd'


class MeasureChannel(PulseChannel):
    """Measure channels transmit measurement stimulus pulses for readout."""
    prefix = 'm'


class ControlChannel(PulseChannel):
    """Control channels provide supplementary control over the qubit to the drive channel.
    These are often associated with multi-qubit gate operations. They may not map trivially
    to a particular qubit index.
    """
    prefix = 'u'


class AcquireChannel(Channel):
    """Acquire channels are used to collect data."""
    prefix = 'a'


class SnapshotChannel(Channel):
    """Snapshot channels are used to specify instructions for simulators."""
    prefix = 's'

    def __init__(self):
        """Create new snapshot channel."""
        super().__init__(0)


class MemorySlot(Channel):
    """Memory slot channels represent classical memory storage."""
    prefix = 'm'


class RegisterSlot(Channel):
    """Classical resister slot channels represent classical registers (low-latency classical
    memory).
    """
    prefix = 'c'
