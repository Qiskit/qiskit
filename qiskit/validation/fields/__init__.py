# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Fields to be used with Qiskit validated classes."""
from datetime import date, datetime

from marshmallow import fields as _fields
from marshmallow.utils import is_collection

from qiskit.validation import ModelValidator
from qiskit.validation.fields.polymorphic import ByAttribute, ByType, TryFrom
from qiskit.validation.fields.containers import Nested, List


__all__ = [
    'Boolean',
    'ByAttribute',
    'ByType',
    'Email',
    'Complex',
    'Date',
    'DateTime',
    'Float',
    'Integer',
    'List',
    'Nested',
    'Number',
    'Raw',
    'String',
    'TryFrom',
    'Url'
]


class Complex(ModelValidator):
    """Field for complex numbers.

    Field for parsing complex numbers:
    * deserializes to Python's `complex`.
    * serializes to a tuple of 2 decimals `(float, imaginary)`
    """

    valid_types = (complex, )

    default_error_messages = {
        'invalid': '{input} cannot be parsed as a complex number.',
        'format': '"{input}" cannot be formatted as complex number.',
    }

    def _serialize(self, value, attr, obj):
        if value is None:
            return None

        if not isinstance(value, complex):
            self.fail('format', input=value)

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


class String(_fields.String, ModelValidator):
    # pylint: disable=missing-docstring
    __doc__ = _fields.String.__doc__

    valid_types = (str, )


class Date(_fields.Date, ModelValidator):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Date.__doc__

    valid_types = (date, )


class DateTime(_fields.DateTime, ModelValidator):
    # pylint: disable=missing-docstring
    __doc__ = _fields.DateTime.__doc__

    valid_types = (datetime, )


class Email(_fields.Email, String):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Email.__doc__


class Url(_fields.Url, String):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Url.__doc__


class Number(_fields.Number, ModelValidator):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Number.__doc__

    valid_types = (int, float)


class Integer(_fields.Integer, ModelValidator):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Integer.__doc__

    valid_types = (int, )


class Float(_fields.Float, ModelValidator):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Float.__doc__

    valid_types = (float, )


class Boolean(_fields.Boolean, ModelValidator):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Boolean.__doc__

    valid_types = (bool, )


class Raw(_fields.Raw, ModelValidator):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Boolean.__doc__
