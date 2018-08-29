# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""QObj validation module for validation against JSON schemas."""

import json
import os
import logging
import jsonschema

from qiskit import QISKitError
from qiskit import __path__ as qiskit_path

_SCHEMAS = {}


logger = logging.getLogger(__name__)


def _load_qobj_schema():
    """Loads the QObj schema for use in future validations."""
    if 'qobj' not in _SCHEMAS:
        sdk = qiskit_path[0]
        # TODO: `transitional_qobj_schema.json` should be replaced with
        # `qobj_schema.json` once we can ensure Qiskit unrollers are emitting
        # valid Qobj structures and the Json Schema extension mechanism is in
        # place.
        schema_file_path = os.path.join(sdk, 'schemas', 'transitional_qobj_schema.json')
        with open(schema_file_path, 'r') as schema_file:
            _SCHEMAS['qobj'] = json.load(schema_file)

    return _SCHEMAS['qobj']


def validate_qobj_against_schema(qobj):
    """Validates a QObj against a schema."""
    qobj_schema = _load_qobj_schema()

    try:
        jsonschema.validate(qobj.as_dict(), qobj_schema)
    except jsonschema.ValidationError as err:
        newerr = QobjValidationError(
            'Qobj failed validation. Set Qiskit log level to DEBUG for further information.')
        newerr.__cause__ = None
        logger.debug('%s', _format_causes(err))
        raise newerr


def _format_causes(err, level=0):
    lines = []

    def _print(string):
        lines.append(_pad(string))

    def _pad(string):
        padding = '  ' * level
        padded_lines = [padding + line for line in string.split('\n')]
        return '\n'.join(padded_lines)

    def _format_path(path):
        def _format(item):
            if isinstance(item, str):
                return '.{}'.format(item)

            return '[{}]'.format(item)

        return ''.join(['<root>'] + list(map(_format, path)))

    _print('\'{}\' failed @ \'{}\' because of:'.format(
        err.validator, _format_path(err.absolute_path)))

    if not err.context:
        _print(str(err.message))
    else:
        for suberr in err.context:
            lines.append(_format_causes(suberr, level+1))

    return '\n'.join(lines)


class QobjValidationError(QISKitError):
    """Represents an error during Qobj validation."""
    pass
