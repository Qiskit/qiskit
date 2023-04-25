# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Qobj utilities and enums."""

from enum import Enum, IntEnum


class QobjType(str, Enum):
    """Qobj.type allowed values."""

    QASM = "QASM"
    PULSE = "PULSE"


class MeasReturnType(str, Enum):
    """PulseQobjConfig meas_return allowed values."""

    AVERAGE = "avg"
    SINGLE = "single"


class MeasLevel(IntEnum):
    """MeasLevel allowed values."""

    RAW = 0
    KERNELED = 1
    CLASSIFIED = 2
