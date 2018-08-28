# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""QObj validation module for validation against JSON schemas."""

import json
import os
import jsonschema

from qiskit import QISKitError
from qiskit import __path__ as qiskit_path

_SCHEMAS = {}


def _load_qobj_schema():
    """Loads the QObj schema for use in future validations.
    """
    if 'qobj' not in _SCHEMAS:
        sdk = qiskit_path[0]
        # Schemas path:     qiskit/backends/schemas
        schemas_path = os.path.join(sdk, 'schemas')
        schema_file_path = os.path.join(schemas_path, 'qobj_schema.json')
        with open(schema_file_path, 'r') as schema_file:
            _SCHEMAS['qobj'] = json.load(schema_file)
    return _SCHEMAS['qobj']


def validate_qobj_against_schema(qobj):
    """Validates a QObj against a schema.
    """
    qobj_schema = _load_qobj_schema()

    # verify the QObj is valid against the schema
    try:
        jsonschema.validate(qobj.as_dict(), qobj_schema)
    except jsonschema.ValidationError as validation_error:
        raise QISKitError(str(validation_error))
