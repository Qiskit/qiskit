# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Snapshot.
"""

from qiskit.pulse.channels import SnapshotChannel
from qiskit.pulse.common.timeslots import TimeslotCollection
from .instruction import Instruction
from .pulse_command import PulseCommand


class Snapshot(PulseCommand, Instruction):
    """Snapshot."""

    def __init__(self, label: str, snap_type: str, start_time: int = 0):
        """Create new snapshot command.

        Args:
            label (str): Snapshot label which is used to identify the snapshot in the output.
            snap_type (str): Type of snapshot, e.g., “state” (take a snapshot of the quantum state).
                The types of snapshots offered are defined in a separate specification
                document for simulators.
            start_time (int, optional): Begin time of snapshot. Defaults to 0.
        """
        PulseCommand.__init__(self, duration=0)
        Instruction.__init__(self, self, start_time, TimeslotCollection([]))
        self._label = label
        self._type = snap_type
        self._channel = SnapshotChannel()

    @property
    def label(self) -> str:
        """Label of snapshot to identify the snapshot."""
        return self._label

    @property
    def type(self) -> str:
        """Type of snapshot."""
        return self._type

    @property
    def channel(self) -> SnapshotChannel:
        """Snapshot channel. """
        return self._channel

    def __eq__(self, other):
        """Two Snapshots are the same if they are of the same type
        and have the same label and type.

        Args:
            other (Snapshot): other Snapshot,

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self._label == other._label and \
                self._type == other._type:
            return True
        return False

    def __repr__(self):
        return '%4d: %s(%s, %s) -> %s' % \
               (self._start_time, self.__class__.__name__, self._label, self._type, self._channel)
