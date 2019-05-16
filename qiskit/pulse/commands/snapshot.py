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
Snapshot.
"""

from qiskit.pulse.channels import SnapshotChannel
from .instruction import Instruction
from .command import Command


class Snapshot(Command, Instruction):
    """Snapshot."""

    def __init__(self, name: str, snap_type: str):
        """Create new snapshot command.

        Args:
            name (str): Snapshot name which is used to identify the snapshot in the output.
            snap_type (str): Type of snapshot, e.g., “state” (take a snapshot of the quantum state).
                The types of snapshots offered are defined in a separate specification
                document for simulators.
        """
        self._type = snap_type
        self._channel = SnapshotChannel()
        Command.__init__(self, duration=0, name=name)
        Instruction.__init__(self, self, self._channel, name=name)
        self._buffer = 0

    @property
    def type(self) -> str:
        """Type of snapshot."""
        return self._type

    def __eq__(self, other):
        """Two Snapshots are the same if they are of the same type
        and have the same name and type.

        Args:
            other (Snapshot): other Snapshot,

        Returns:
            bool: are self and other equal.
        """
        if (type(self) is type(other) and
                self.name == other.name and
                self.type == other.type):
            return True
        return False

    # pylint: disable=arguments-differ
    def to_instruction(self):
        return self
    # pylint: enable=arguments-differ

    def __repr__(self):
        return '%s(%s, %s) -> %s' % (self.__class__.__name__, self.name,
                                     self.type, self.channels)
