# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Snapshot.
"""

from qiskit.pulse.channels import SnapshotChannel
from qiskit.pulse.common.interfaces import Instruction
from qiskit.pulse.common.timeslots import TimeslotOccupancy
from .pulse_command import PulseCommand


class Snapshot(PulseCommand, Instruction):
    """Snapshot."""

    def __init__(self, label, snap_type):
        """Create new snapshot command.

        Args:
            label (str): Snapshot label which is used to identify the snapshot in the output.
            snap_type (str): Type of snapshot, e.g., “state” (take a snapshot of the quantum state).
                The types of snapshots offered are defined in a separate specification
                document for simulators.
        """
        super().__init__(duration=0)
        self.label = label
        self.type = snap_type
        self._channel = SnapshotChannel()
        self._occupancy = TimeslotOccupancy([])

    def __eq__(self, other):
        """Two Snapshots are the same if they are of the same type
        and have the same label and type.

        Args:
            other (Snapshot): other Snapshot,

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self.label == other.label and \
                self.type == other.type:
            return True
        return False

    @property
    def duration(self):
        return 0

    @property
    def occupancy(self):
        return self._occupancy

    @property
    def command(self) -> 'Snapshot':
        """Snapshot command. """
        return self

    @property
    def channel(self) -> SnapshotChannel:
        """Snapshot channel. """
        return self._channel

    def __repr__(self):
        return '%s(%s, %s) >> %s' % (self.__class__.__name__, self.label, self.type, self._channel)
