# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A simulator instruction to capture output within a simulation. The types of snapshot
instructions available are determined by the simulator being used.
"""
from typing import Optional

from ..channels import SnapshotChannel
from .instruction import Instruction


class Snapshot(Instruction):
    """An instruction targeted for simulators, to capture a moment in the simulation."""

    def __init__(self, label: str, snapshot_type: str = 'statevector', name: Optional[str] = None):
        """Create new snapshot.

        Args:
            label: Snapshot label which is used to identify the snapshot in the output.
            snapshot_type: Type of snapshot, e.g., “state” (take a snapshot of the quantum state).
                           The types of snapshots offered are defined by the simulator used.
            name: Snapshot name which defaults to ``label``. This parameter is only for display
                  purposes and is not taken into account during comparison.
        """
        self._label = label
        self._type = snapshot_type
        self._channel = SnapshotChannel()
        if name is None:
            name = self.label
        super().__init__((label, snapshot_type), 0, (self.channel,), name=name)

    @property
    def label(self) -> str:
        """Label of snapshot."""
        return self._label

    @property
    def type(self) -> str:
        """Type of snapshot."""
        return self._type

    @property
    def channel(self) -> SnapshotChannel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on; trivially, a ``SnapshotChannel``.
        """
        return self._channel
