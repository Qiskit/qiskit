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


# TODO: According to [1], this will become the infrastructure for all the
# validation procedures. It should be generalized when extending the validation
# functionality according to the ideas described in [1] and [2].
#
# [1] https://github.com/Qiskit/qiskit-terra/pull/807/files#r214653924
# [2] https://github.com/Qiskit/qiskit-terra/pull/807/files#r214055642

def _load_qobj_schema():
    """Loads the QObj schema for use in future validations."""
    if 'qobj' not in _SCHEMAS:
        sdk = qiskit_path[0]
        # TODO: `transitional_qobj_schema.json` should be replaced with
        # `qobj_schema.json` once we can ensure Qiskit unrollers are emitting
        # valid Qobj structures and the Json Schema extension mechanism is in
        # place.
        schema_file_path = os.path.join(sdk, 'schemas', 'qobj_schema.json')
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
        newerr.__cause__ = _SummaryValidationError(err)
        logger.debug('%s', _format_causes(err))
        raise newerr


def _format_causes(err, level=0):
    """Return a cascading explanation of the validation error in the form
    of::

        <validator> failed @ <subfield_path> because of:
            <validator> failed @ <subfield_path> because of:
                ...
            <validator> failed @ <subfield_path> because of:
                ...
            ...

    For example::

        'oneOf' failed @ '<root>' because of:
            'required' failed @ '<root>.config' because of:
                'meas_level' is a required property

    Meaning the validator 'oneOf' failed while validating the whole object
    because of the validator 'required' failing while validating the property
    'config' because its 'meas_level' field is missing.

    The cascade repeats the format "<validator> failed @ <path> because of"
    until there are no deeper causes. In this case, the string representation
    of the error is shown.

    Args:
        err (jsonschema.ValidationError): the instance to explain.
        level (int): starting level of indentation for the cascade of
            explanations.

    Return:
        str: a formatted string with the explanation of the error.

    """
    lines = []

    def _print(string, offset=0):
        lines.append(_pad(string, offset=offset))

    def _pad(string, offset=0):
        padding = '  ' * (level + offset)
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
        _print(str(err.message), offset=1)
    else:
        for suberr in err.context:
            lines.append(_format_causes(suberr, level+1))

    return '\n'.join(lines)


class QobjValidationError(QISKitError):
    """Represents an error during Qobj validation."""
    pass


class _SummaryValidationError(QISKitError):
    """Cut off the message of a jsonschema.ValidationError to avoid printing
    noise in the standard output. The original validation error is in the
    `validation_error` property.

    Attributes:
        validation_error (jsonschama.ValidationError): original validations
            error.
    """

    def __init__(self, validation_error):
        super().__init__(self._shorten_message(str(validation_error)))
        self.validation_error = validation_error

    def _shorten_message(self, message):
        if len(message) > 1000:
            return 'Original message too long to be useful: {}[...]'.format(message[:1000])

        return message
