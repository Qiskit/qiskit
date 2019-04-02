# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Qobj utilities and enums."""

from enum import Enum

from qiskit.validation.jsonschema import validate_json_against_schema


class QobjType(str, Enum):
    """Qobj.type allowed values."""
    QASM = 'QASM'
    PULSE = 'PULSE'


class MeasReturnType(str, Enum):
    """PulseQobjConfig meas_return allowed values."""
    AVERAGE = 'avg'
    SINGLE = 'single'


def validate_qobj_against_schema(qobj):
    """Validates a QObj against the .json schema.

    Args:
        qobj (Qobj): Qobj to be validated.
    """
    validate_json_against_schema(
        qobj.as_dict(), 'qobj',
        err_msg='Qobj failed validation. Set Qiskit log level to DEBUG '
                'for further information.')
