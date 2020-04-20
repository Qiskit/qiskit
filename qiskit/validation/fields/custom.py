# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Fields custom to Terra to be used with Qiskit validated classes."""

from collections.abc import Mapping

import numpy

from marshmallow.utils import is_collection
from marshmallow.exceptions import ValidationError

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.validation import ModelTypeValidator


class Complex(ModelTypeValidator):
    """Field for complex numbers.

    Field for parsing complex numbers:
    * deserializes from either a Python `complex` or 2-element collection.
    * serializes to a tuple of 2 decimals `(real, imaginary)`
    """

    valid_types = (complex, )

    default_error_messages = {
        'invalid': '{input} cannot be parsed as a complex number.',
        'format': '"{input}" cannot be formatted as complex number.',
    }

    def _serialize(self, value, attr, obj, **_):
        try:
            return [value.real, value.imag]
        except AttributeError:
            raise self.make_error_serialize('format', input=value)

    def _deserialize(self, value, attr, data, **_):
        if is_collection(value) and (len(value) == 2):
            try:
                return complex(value[0], value[1])
            except (ValueError, TypeError):
                raise self.make_error('invalid', input=value)
        elif isinstance(value, complex):
            return value

        raise self.make_error('invalid', input=value)


class InstructionParameter(ModelTypeValidator):
    """Field for objects used in instruction parameters.

    This field provides support for parsing objects of types that uses by
    qobj.experiments.instructions.parameters:
    * basic Python types: complex, int, float, str, list
    * ``numpy``: integer, float, ndarray

    Note that by using this field, serialization-deserialization round-tripping
    becomes not possible, as certain types serialize to the same Python basic
    type (for example, numpy.float and regular float). If possible, it is
    recommended that more specific and defined fields are used instead.
    """
    valid_types = (complex, int, float, str,
                   ParameterExpression,
                   numpy.integer, numpy.float,
                   list, numpy.ndarray)

    default_error_messages = {
        'invalid': '{input} cannot be parsed as a parameter.',
        'format': '"{input}" cannot be formatted as a parameter.'
    }

    def _serialize(self, value, attr, obj, **kwargs):
        # pylint: disable=too-many-return-statements
        if is_collection(value):
            return [self._serialize(item, attr, obj, **kwargs) for item in value]

        if isinstance(value, complex):
            return [value.real, value.imag]
        if isinstance(value, numpy.integer):
            return int(value)
        if isinstance(value, numpy.float):
            return float(value)
        if isinstance(value, (float, int, str)):
            return value
        if isinstance(value, ParameterExpression):
            if value.parameters:
                raise self.make_error_serialize('invalid', input=value)
            return float(value)

        # Fallback for attempting serialization.
        if hasattr(value, 'to_dict'):
            return value.to_dict()

        raise self.make_error_serialize('format', input=value)

    def _deserialize(self, value, attr, data, **kwargs):
        if is_collection(value):
            return [self._deserialize(item, attr, data, **kwargs) for item in value]

        if isinstance(value, (float, int, str)):
            return value

        raise self.make_error('invalid', input=value)

    def check_type(self, value, attr, data, **kwargs):
        """Customize check_type for handling containers."""
        # Check the type in the standard way first, in order to fail quickly
        # in case of invalid values.
        root_value = super().check_type(value, attr, data, **kwargs)

        if is_collection(value):
            _ = [super(InstructionParameter, self).check_type(item, attr, data, **kwargs)
                 for item in value]

        return root_value


class DictParameters(ModelTypeValidator):
    """Field for objects used in measurement kernel and discriminator parameters.
    """
    default_error_messages = {
        'invalid_mapping': 'Not a valid mapping type.',
        'invalid': '{input} cannot be parsed as a parameter.'
    }

    def __init__(self, valid_value_types, **kwargs):
        """Create new model.

        Args:
            valid_value_types (tuple): valid types as values.
        """
        # pylint: disable=missing-param-doc

        super().__init__(**kwargs)
        self.valid_value_types = valid_value_types

    def _expected_types(self):
        return self.valid_value_types

    def check_type(self, value, attr, data, **kwargs):
        if value is None:
            return None

        _check_type = super().check_type

        errors = []
        if not isinstance(data[attr], Mapping):
            raise self.make_error('invalid_mapping')

        try:
            if isinstance(value, Mapping):
                for v in value.values():
                    self.check_type(v, attr, data, **kwargs)
            elif is_collection(value):
                for v in value:
                    self.check_type(v, attr, data, **kwargs)
            else:
                _check_type(value, attr, data, **kwargs)
        except ValidationError as err:
            errors.append(err.messages)

        if errors:
            raise ValidationError(errors)

        return value

    def _validate_values(self, value):
        if value is None:
            return None
        if isinstance(value, complex):
            return [value.real, value.imag]
        if isinstance(value, self.valid_value_types):
            return value
        if is_collection(value):
            return [self._validate_values(each) for each in value]
        if isinstance(value, Mapping):
            return {str(k): self._validate_values(v) for k, v in value.items()}

        raise self.make_error('invalid', input=value)

    def _serialize(self, value, attr, obj, **_):
        if value is None:
            return None
        if isinstance(value, Mapping):
            return {str(k): self._validate_values(v) for k, v in value.items()}

        raise self.make_error_serialize('invalid_mapping')

    def _deserialize(self, value, attr, data, **_):
        if value is None:
            return None
        if isinstance(value, Mapping):
            return value

        raise self.make_error('invalid_mapping')
