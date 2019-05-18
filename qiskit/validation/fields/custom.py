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

import numpy
import sympy

from marshmallow.utils import is_collection
from marshmallow.exceptions import ValidationError
from marshmallow.compat import Mapping

from qiskit.validation import ModelTypeValidator


class Complex(ModelTypeValidator):
    """Field for complex numbers.

    Field for parsing complex numbers:
    * deserializes to Python's `complex`.
    * serializes to a tuple of 2 decimals `(real, imaginary)`
    """

    valid_types = (complex, )

    default_error_messages = {
        'invalid': '{input} cannot be parsed as a complex number.',
        'format': '"{input}" cannot be formatted as complex number.',
    }

    def _serialize(self, value, attr, obj):
        try:
            return [value.real, value.imag]
        except AttributeError:
            self.fail('format', input=value)

    def _deserialize(self, value, attr, data):
        if not is_collection(value) or len(value) != 2:
            self.fail('invalid', input=value)

        try:
            return complex(*value)
        except (ValueError, TypeError):
            self.fail('invalid', input=value)


class InstructionParameter(ModelTypeValidator):
    """Field for objects used in instruction parameters.

    This field provides support for parsing objects of types that uses by
    qobj.experiments.instructions.parameters:
    * basic Python types: complex, int, float, str, list
    * ``numpy``: integer, float, ndarray
    * ``sympy``: Symbol, Basic

    Note that by using this field, serialization-deserialization round-tripping
    becomes not possible, as certain types serialize to the same Python basic
    type (for example, numpy.float and regular float). If possible, it is
    recommended that more specific and defined fields are used instead.
    """

    valid_types = (complex, int, float, str,
                   numpy.integer, numpy.float, sympy.Basic, sympy.Symbol,
                   list, numpy.ndarray)

    default_error_messages = {
        'invalid': '{input} cannot be parsed as a parameter.',
        'format': '"{input}" cannot be formatted as a parameter.'
    }

    def _serialize(self, value, attr, obj):
        # pylint: disable=too-many-return-statements
        if is_collection(value):
            return [self._serialize(item, attr, obj) for item in value]

        if isinstance(value, complex):
            return [value.real, value.imag]
        if isinstance(value, numpy.integer):
            return int(value)
        if isinstance(value, numpy.float):
            return float(value)
        if isinstance(value, (float, int, str)):
            return value
        if isinstance(value, sympy.Symbol):
            return str(value)
        if isinstance(value, sympy.Basic):
            if value.is_imaginary:
                return [float(sympy.re(value)), float(sympy.im(value))]
            if value.is_Integer:
                return int(value.evalf())
            else:
                return float(value.evalf())

        # Fallback for attempting serialization.
        if hasattr(value, 'as_dict'):
            return value.as_dict()

        return self.fail('format', input=value)

    def _deserialize(self, value, attr, data):
        if is_collection(value):
            return [self._deserialize(item, attr, data) for item in value]

        if isinstance(value, (float, int, str)):
            return value

        return self.fail('invalid', input=value)

    def check_type(self, value, attr, data):
        """Customize check_type for handling containers."""
        # Check the type in the standard way first, in order to fail quickly
        # in case of invalid values.
        root_value = super().check_type(
            value, attr, data)

        if is_collection(value):
            _ = [super(InstructionParameter, self).check_type(item, attr, data)
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

    def check_type(self, value, attr, data):
        if value is None:
            return None

        _check_type = super().check_type

        errors = []
        if not isinstance(data[attr], Mapping):
            self.fail('invalid_mapping')

        try:
            if isinstance(value, Mapping):
                for v in value.values():
                    self.check_type(v, attr, data)
            elif is_collection(value):
                for v in value:
                    self.check_type(v, attr, data)
            else:
                _check_type(value, attr, data)
        except ValidationError as err:
            errors.append(err.messages)

        if errors:
            raise ValidationError(errors)

        return value

    def _validate_values(self, value):
        if value is None:
            return None
        if isinstance(value, self.valid_value_types):
            return value
        if is_collection(value):
            return [self._validate_values(each) for each in value]
        if isinstance(value, Mapping):
            return {str(k): self._validate_values(v) for k, v in value.items()}

        return self.fail('invalid', input=value)

    def _serialize(self, value, attr, obj):
        if value is None:
            return None
        if isinstance(value, Mapping):
            return {str(k): self._validate_values(v) for k, v in value.items()}

        return self.fail('invalid_mapping')

    def _deserialize(self, value, attr, data):
        if value is None:
            return None
        if isinstance(value, Mapping):
            return value

        return self.fail('invalid_mapping')
