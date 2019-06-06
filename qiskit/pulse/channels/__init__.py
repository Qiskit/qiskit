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

"""Device-related classes for pulse."""

from .device_specification import DeviceSpecification
from .pulse_channels import DriveChannel, ControlChannel, MeasureChannel
from .pulse_channels import PulseChannel
from .channels import AcquireChannel, MemorySlot, RegisterSlot, SnapshotChannel
from .channels import Channel
from .qubit import Qubit
