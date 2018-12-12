# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Container fields that represent nested/collections of schemas or types."""

from collections.abc import Iterable

from marshmallow import fields as _fields
from marshmallow.utils import is_collection

from qiskit.validation import ValidationError, ModelTypeValidator


class Nested(_fields.Nested, ModelTypeValidator):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Nested.__doc__

    def _expected_types(self):
        return self.schema.model_cls

    def check_type(self, value, attr, data):
        """Validate if the value is of the type of the schema's model.

        Assumes the nested schema is a ``BaseSchema``.
        """
        if self.many and not is_collection(value):
            raise self._not_expected_type(
                value, Iterable, fields=[self], field_names=attr, data=data)

        _check_type = super().check_type

        errors = []
        values = value if self.many else [value]
        for idx, v in enumerate(values):
            try:
                _check_type(v, idx, values)
            except ValidationError as err:
                errors.append(err.messages)

        if errors:
            errors = errors if self.many else errors[0]
            raise ValidationError(errors)

        return value


class List(_fields.List, ModelTypeValidator):
    # pylint: disable=missing-docstring
    __doc__ = _fields.List.__doc__

    valid_types = (Iterable, )

    def check_type(self, value, attr, data):
        """Validate if it's a list of valid item-field values.

        Check if each element in the list can be validated by the item-field
        passed during construction.
        """
        super().check_type(value, attr, data)

        errors = []
        for idx, v in enumerate(value):
            try:
                self.container.check_type(v, idx, value)
            except ValidationError as err:
                errors.append(err.messages)

        if errors:
            raise ValidationError(errors)

        return value
