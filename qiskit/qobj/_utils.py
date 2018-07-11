# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Qobj utilities and enums."""

from enum import Enum

from qiskit import QISKitError


class QobjType(str, Enum):
    """
    Qobj.type allowed values.
    """
    QASM = 'QASM'
    PULSE = 'PULSE'


class QobjValidationError(QISKitError):
    """
    Represents an error during Qobj validation.
    """
    pass
