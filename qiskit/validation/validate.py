# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Validators for Qiskit validated classes."""

from marshmallow import ValidationError
from marshmallow.validate import Validator


class Or(Validator):
    """Validate using a boolean "or" against a list of Validators.

    This validator accepts a list of other ``Validators``, and returns True if
    any of those validators pass.

    Examples:

        wheels = Integer(validate=Or(Equal(4), Equal(2)))
    """

    def __init__(self, validators):
        """
        Args:
            validators (list[Validator]): list of Validators.
        """
        self.validators = validators

    def __call__(self, value):
        for validator in self.validators:
            try:
                return validator(value)
            except ValidationError:
                pass

        raise ValidationError('Data could not be validated against any '
                              'validator')


class PatternProperties(Validator):
    """Validate the keys and values of an object, disallowing additional ones.

    This validator is a combination of the jsonschema `patternProperties` and
    `additionalProperties == False`. It enforces that the keys of an object
    conform to any of the specified validators in the mapping, and its value
    has the specified type.

    Examples::

        counts = Nested(
            SomeSchema,
            validate=PatternProperties(
                {Regexp('^0x([0-9A-Fa-f])+$'): Integer(),
                 Regexp('OTHER_THING'): String()})

    """

    def __init__(self, pattern_properties):
        """
        Args:
            pattern_properties (dict[Validator: Field]): dictionary of the
                valid mappings.
        """
        self.pattern_properties = pattern_properties

    def __call__(self, value):
        errors = {}
        for key, value_ in value.__dict__.items():
            # Attempt to validate the keys against any field.
            field = None
            for validator, candidate in self.pattern_properties.items():
                try:
                    validator(key)
                    field = candidate
                except ValidationError as ex:
                    errors[key] = ex.messages

            # Attempt to validate the contents.
            if field:
                errors.pop(key, None)
                try:
                    field.deserialize(value_)
                except ValidationError as ex:
                    errors[key] = ex.messages

        if errors:
            raise ValidationError(errors)

        return value
