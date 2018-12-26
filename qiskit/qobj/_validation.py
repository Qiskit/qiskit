# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""QObj validation module for validation against JSON schemas."""

from . import _schema_validation


def validate_qobj_against_schema(qobj):
    """Validates a QObj against a schema."""
    _schema_validation.validate_json_against_schema(
        qobj.as_dict(), 'qobj',
        err_msg='Qobj failed validation. '
                'Set Qiskit log level to DEBUG '
                'for further information.')
