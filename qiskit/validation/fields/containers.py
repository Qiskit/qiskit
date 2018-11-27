# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Container fields are those that represent nested/collections of schemas or
types."""

from marshmallow import fields as _fields
from marshmallow.utils import is_collection

from qiskit.validation import ValidationError, ModelValidator


class Nested(_fields.Nested, ModelValidator):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Nested.__doc__

    def _expected_types(self):
        return self.schema.model_cls


class List(_fields.List, ModelValidator):
    # pylint: disable=missing-docstring
    __doc__ = _fields.List.__doc__

    def validate_model(self, value, *_):
        # pylint: disable=arguments-differ
        errors = []
        if not is_collection(value):
            raise self._not_expected_type(value, 'iterable')

        for idx, v in enumerate(value):
            if isinstance(v, ModelValidator):
                try:
                    self.container.validate_model(v, idx, value)
                except ValidationError as err:
                    errors.append(err.messages)

        if errors:
            raise ValidationError(errors)

        return value
