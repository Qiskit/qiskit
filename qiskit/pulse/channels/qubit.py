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
Physical qubit.
"""
from typing import Tuple

from .pulse_channels import DriveChannel, ControlChannel, MeasureChannel
from .channels import AcquireChannel


class Qubit:
    """Physical qubit."""

    def __init__(self, index: int,
                 drive_channel: DriveChannel,
                 measure_channel: MeasureChannel,
                 acquire_channel: AcquireChannel,
                 control_channels: Tuple[ControlChannel] = None):
        self._index = index
        self._drive = drive_channel
        self._controls = tuple(control_channels) if control_channels else tuple()
        self._measure = measure_channel
        self._acquire = acquire_channel

    @property
    def index(self) -> int:
        """Return the index of this qubit."""
        return self._index

    @property
    def drive(self) -> DriveChannel:
        """Return the drive channel of this qubit."""
        return self._drive

    @property
    def measure(self) -> MeasureChannel:
        """Return the measure channel of this qubit."""
        return self._measure

    @property
    def acquire(self) -> AcquireChannel:
        """Return the primary acquire channel of this qubit."""
        return self._acquire

    @property
    def controls(self) -> Tuple[ControlChannel]:
        """Return the control channels for this qubit."""
        return self._controls

    def __eq__(self, other):
        """Two physical qubits are the same if they have the same index and channels.

        Args:
            other (Qubit): other Qubit

        Returns:
            bool: are self and other equal.
        """
        # pylint: disable=too-many-boolean-expressions
        if (type(self) is type(other) and
                self.index == other.index and
                self.drive == other.drive and
                self.measure == other.measure and
                self.acquire == other.acquire and
                self.controls == other.controls):
            return True
        return False
