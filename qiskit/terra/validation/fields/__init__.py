# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Fields to be used with Qiskit validated classes.

When extending this module with new Fields:

    1. Distinguish a new type, like the ``Complex`` number in this module.
    2. Use a new Marshmallow field not used in ``qiskit`` yet.

Marshamallow fields does not allow model validation so you need to create a new
field, make it subclass of the Marshamallow field *and* ``ModelTypeValidator``,
and redefine ``valid_types`` to be the list of valid types. Usually, **the
same types this field deserializes to**. For instance::

    class Boolean(marshmallow.fields.Boolean, ModelTypeValidator):
        __doc__ = _fields.Boolean.__doc__

        valid_types = (bool, )

See ``ModelTypeValidator`` for more subclassing options.
"""
from datetime import date, datetime

from marshmallow import fields as _fields
from marshmallow.utils import is_collection

from qiskit.validation import ModelTypeValidator
from qiskit.validation.fields.polymorphic import ByAttribute, ByType, TryFrom
from qiskit.validation.fields.containers import Nested, List


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


class String(_fields.String, ModelTypeValidator):
    # pylint: disable=missing-docstring
    __doc__ = _fields.String.__doc__

    valid_types = (str, )


class Date(_fields.Date, ModelTypeValidator):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Date.__doc__

    valid_types = (date, )


class DateTime(_fields.DateTime, ModelTypeValidator):
    # pylint: disable=missing-docstring
    __doc__ = _fields.DateTime.__doc__

    valid_types = (datetime, )


class Email(_fields.Email, String):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Email.__doc__


class Url(_fields.Url, String):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Url.__doc__


class Number(_fields.Number, ModelTypeValidator):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Number.__doc__

    def _expected_types(self):
        return self.num_type


class Integer(_fields.Integer, Number):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Integer.__doc__


class Float(_fields.Float, Number):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Float.__doc__


class Boolean(_fields.Boolean, ModelTypeValidator):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Boolean.__doc__

    valid_types = (bool, )


class Raw(_fields.Raw, ModelTypeValidator):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Boolean.__doc__
