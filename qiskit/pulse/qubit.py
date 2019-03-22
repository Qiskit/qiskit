# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Physical qubit.
"""
from typing import List

from qiskit.pulse.channels import AcquireChannel
from qiskit.pulse.channels import DriveChannel, ControlChannel, MeasureChannel
from qiskit.pulse.exceptions import PulseError


class Qubit:
    """Physical qubit."""

    def __init__(self, index: int,
                 drive_channels: List[DriveChannel] = None,
                 control_channels: List[ControlChannel] = None,
                 measure_channels: List[MeasureChannel] = None,
                 acquire_channels: List[AcquireChannel] = None):
        self._index = index
        self._drives = drive_channels
        self._controls = control_channels
        self._measures = measure_channels
        self._acquires = acquire_channels

    def __eq__(self, other):
        """Two physical qubits are the same if they have the same index and channels.

        Args:
            other (Qubit): other Qubit

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self._index == other._index and \
                self._drives == other._drives and \
                self._controls == other._controls and \
                self._measures == other._measures and \
                self._acquires == other._acquires:
            return True
        return False

    @property
    def drive(self):
        if self._drives:
            return self._drives[0]
        else:
            raise PulseError("No drive channels in q[%d]" % self._index)

    @property
    def control(self):
        if self._controls:
            return self._controls[0]
        else:
            raise PulseError("No control channels in q[%d]" % self._index)

    @property
    def measure(self):
        if self._measures:
            return self._measures[0]
        else:
            raise PulseError("No measurement channels in q[%d]" % self._index)

    @property
    def acquire(self):
        if self._acquires:
            return self._acquires[0]
        else:
            raise PulseError("No acquire channels in q[%d]" % self._index)
