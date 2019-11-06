# -*- coding: utf-8 -*-

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

"""Validation module for validation against JSON schemas."""

import json
import os
import logging
import jsonschema

from .exceptions import SchemaValidationError, _SummaryValidationError

logger = logging.getLogger(__name__)


_DEFAULT_SCHEMA_PATHS = {
    'backend_configuration': 'schemas/backend_configuration_schema.json',
    'backend_properties': 'schemas/backend_properties_schema.json',
    'backend_status': 'schemas/backend_status_schema.json',
    'default_pulse_configuration': 'schemas/default_pulse_configuration_schema.json',
    'job_status': 'schemas/job_status_schema.json',
    'qobj': 'schemas/qobj_schema.json',
    'result': 'schemas/result_schema.json'}
# Schema and Validator storage
_SCHEMAS = {}
_VALIDATORS = {}


def _load_schema(file_path, name=None):
    """Loads the QObj schema for use in future validations.

   Caches schema in _SCHEMAS module attribute.

   Args:
        file_path(str): Path to schema.
        name(str): Given name for schema. Defaults to file_path filename
            without schema.
   Return:
        schema(dict): Loaded schema.
    """
    if name is None:
        # filename without extension
        name = os.path.splitext(os.path.basename(file_path))[0]
    if name not in _SCHEMAS:
        with open(file_path, 'r') as schema_file:
            _SCHEMAS[name] = json.load(schema_file)

    return _SCHEMAS[name]


def _get_validator(name, schema=None, check_schema=True,
                   validator_class=None, **validator_kwargs):
    """Generate validator for JSON schema.

    Args:
        name (str): Name for validator. Will be validator key in
            `_VALIDATORS` dict.
        schema (dict): JSON schema `dict`. If not provided searches for schema
            in `_SCHEMAS`.
        check_schema (bool): Verify schema is valid.
        validator_class (jsonschema.IValidator): jsonschema IValidator instance.
            Default behavior is to determine this from the schema `$schema`
            field.
        **validator_kwargs: Additional keyword arguments for validator.

    Return:
        jsonschema.IValidator: Validator for JSON schema.

    Raises:
        SchemaValidationError: Raised if validation fails.
    """
    if schema is None:
        try:
            schema = _SCHEMAS[name]
        except KeyError:
            raise SchemaValidationError("Valid schema name or schema must "
                                        "be provided.")

    if name not in _VALIDATORS:
        # Resolve JSON spec from schema if needed
        if validator_class is None:
            validator_class = jsonschema.validators.validator_for(schema)

        # Generate and store validator in _VALIDATORS
        _VALIDATORS[name] = validator_class(schema, **validator_kwargs)
        if check_schema:
            _VALIDATORS[name].check_schema(schema)

    validator = _VALIDATORS[name]
    return validator


def _load_schemas_and_validators():
    """Load all default schemas into `_SCHEMAS`."""
    schema_base_path = os.path.join(os.path.dirname(__file__), '../..')
    for name, path in _DEFAULT_SCHEMA_PATHS.items():
        _load_schema(os.path.join(schema_base_path, path), name)
        _get_validator(name)


# Load all schemas on import
_load_schemas_and_validators()


def validate_json_against_schema(json_dict, schema,
                                 err_msg=None):
    """Validates JSON dict against a schema.

    Args:
        json_dict (dict): JSON to be validated.
        schema (dict or str): JSON schema dictionary or the name of one of the
            standards schemas in Qiskit to validate against it. The list of
            standard schemas is: ``backend_configuration``,
            ``backend_properties``, ``backend_status``,
            ``default_pulse_configuration``, ``job_status``, ``qobj``,
            ``result``.
        err_msg (str): Optional error message.

    Raises:
        SchemaValidationError: Raised if validation fails.
    """

    try:
        if isinstance(schema, str):
            schema_name = schema
            schema = _SCHEMAS[schema_name]
            validator = _get_validator(schema_name)
            validator.validate(json_dict)
        else:
            jsonschema.validate(json_dict, schema)
    except jsonschema.ValidationError as err:
        if err_msg is None:
            err_msg = "JSON failed validation. Set Qiskit log level to DEBUG " \
                      "for further information."
        newerr = SchemaValidationError(err_msg)
        newerr.__cause__ = _SummaryValidationError(err)
        logger.debug('%s', _format_causes(err))
        raise newerr


def _format_causes(err, level=0):
    """Return a cascading explanation of the validation error.

    Returns a cascading explanation of the validation error in the form of::

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
