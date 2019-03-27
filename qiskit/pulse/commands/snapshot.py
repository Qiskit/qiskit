# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Snapshot.
"""
from typing import Set

from qiskit.pulse.channels import PulseChannel, SnapshotChannel
from qiskit.pulse.common.interfaces import Pulse
from qiskit.pulse.common.timeslots import Interval, Timeslot, TimeslotOccupancy
from .pulse_command import PulseCommand


class SnapshotPulse(Pulse):
    """Pulse to keep persistent value. """

    def __init__(self, command: 'Snapshot'):
        self._command = command
        self._channel = SnapshotChannel()
        self._occupancy = TimeslotOccupancy([Timeslot(Interval(0, 0), self._channel)])

    @property
    def duration(self):
        return 0

    @property
    def channelset(self) -> Set[PulseChannel]:
        return {self._channel}

    @property
    def occupancy(self):
        return self._occupancy


class Snapshot(PulseCommand):
    """Snapshot."""

    def __init__(self, label, snap_type):
        """Create new snapshot command.

        Args:
            label (str): Snapshot label which is used to identify the snapshot in the output.
            snap_type (str): Type of snapshot, e.g., “state” (take a snapshot of the quantum state).
                The types of snapshots offered are defined in a separate specification
                document for simulators.
        """

        super(Snapshot, self).__init__(duration=0, name='snapshot')

        self.label = label
        self.type = snap_type

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

    def __call__(self) -> SnapshotPulse:
        return SnapshotPulse(self)
