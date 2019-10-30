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

"""Validators for Qiskit validated classes."""

from collections.abc import Mapping

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
        """Or initializer.

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
        """PatternProperties initializer.

        Args:
            pattern_properties (dict[Validator: Field]): dictionary of the
                valid mappings.
        """
        self.pattern_properties = pattern_properties

    def __call__(self, value):
        errors = {}

        if isinstance(value, Mapping):
            _dict = value
        else:
            _dict = value.__dict__

        for key, value_ in _dict.items():
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
