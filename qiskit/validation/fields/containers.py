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

"""Container fields that represent nested/collections of schemas or types."""

from collections.abc import Iterable, Mapping

import numpy as np

from marshmallow import fields as _fields
from marshmallow.exceptions import ValidationError
from marshmallow.utils import is_collection

from qiskit.validation import ModelTypeValidator


class Nested(_fields.Nested, ModelTypeValidator):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Nested.__doc__

    def _expected_types(self):
        return self.schema.model_cls

    def check_type(self, value, attr, data, **kwargs):
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
                _check_type(v, idx, values, **kwargs)
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

    def check_type(self, value, attr, data, **kwargs):
        """Validate if it's a list of valid item-field values.

        Check if each element in the list can be validated by the item-field
        passed during construction.
        """
        super().check_type(value, attr, data, **kwargs)

        errors = []
        for idx, v in enumerate(value):
            try:
                self.inner.check_type(v, idx, value, **kwargs)
            except ValidationError as err:
                errors.append(err.messages)

        if errors:
            raise ValidationError(errors)

        return value


class Dict(_fields.Dict, ModelTypeValidator):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Dict.__doc__

    valid_types = (Mapping, )


class NumpyArray(List):
    # pylint: disable=missing-docstring
    __doc__ = List.__doc__

    def _deserialize(self, value, attr, data, **kwargs):
        # If an numpy array just return that:
        if isinstance(value, np.ndarray):
            return value
        # If not a native numpy array deserialize the list and convert:
        deserialized_list = super(NumpyArray, self)._deserialize(
            value, attr, data, **kwargs)
        try:
            return np.array(deserialized_list)
        except ValueError as err:
            raise ValidationError([err])
