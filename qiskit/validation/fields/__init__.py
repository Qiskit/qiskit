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

"""Fields to be used with Qiskit validated classes.

When extending this module with new Fields:

    1. Distinguish a new type, like the ``Complex`` number in this module.
    2. Use a new Marshmallow field not used in ``qiskit`` yet.

Marshmallow fields does not allow model validation so you need to create a new
field, make it subclass of the Marshmallow field *and* ``ModelTypeValidator``,
and redefine ``valid_types`` to be the list of valid types. Usually, **the
same types this field deserializes to**. For instance::

    class Boolean(marshmallow.fields.Boolean, ModelTypeValidator):
        __doc__ = _fields.Boolean.__doc__

        valid_types = (bool, )

See ``ModelTypeValidator`` for more subclassing options.
"""

from datetime import date, datetime

from marshmallow import fields as _fields

from qiskit.validation import ModelTypeValidator
from qiskit.validation.fields.polymorphic import ByAttribute, ByType, TryFrom
from qiskit.validation.fields.containers import Nested, List, Dict, NumpyArray

from .custom import Complex, InstructionParameter, DictParameters


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
