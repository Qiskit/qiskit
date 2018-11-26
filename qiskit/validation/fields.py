# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Fields to be used with Qiskit validated classes."""
from functools import partial
from datetime import date, datetime

from marshmallow import fields as _fields
from marshmallow.fields import Raw
from marshmallow.utils import is_collection
from marshmallow_polyfield import PolyField

from . import ValidationError


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


class BasePolyField(PolyField):
    """Base class for polymorphic fields.

    Defines a Field that can contain data of different types. Deciding the
    type is performed by the ``to_dict_selector()`` and ``from_dict_selector()``
    functions, that act on ``choices``. Subclasses are recommended to:

    * define the type of the ``choices`` attribute. It should contain a
      reference to the individual Schemas that are accepted by the field, along
      with other information specific to the subclass.
    * customize the ``to_dict_selector()`` and ``from_dict_selector()``, adding
      the necessary logic for inspecting ``choices`` and the data, and
      returning one of the Schemas.

     Args:
        choices (iterable): iterable containing the schema instances and the
            information needed for performing disambiguation.
        many (bool): whether the field is a collection of objects.
        metadata (dict): the same keyword arguments that ``PolyField`` receives.
    """

    def __init__(self, choices, many=False, **metadata):
        to_dict_selector = partial(self.to_dict_selector, choices)
        from_dict_selector = partial(self.from_dict_selector, choices)

        super().__init__(to_dict_selector, from_dict_selector, many=many, **metadata)

    def to_dict_selector(self, choices, *args, **kwargs):
        """Return an schema in `choices` for serialization."""
        raise NotImplementedError

    def from_dict_selector(self, choices, *args, **kwargs):
        """Return an schema in `choices` for deserialization."""
        raise NotImplementedError

    def _deserialize(self, value, attr, data):
        """Override _deserialize for customizing the Exception raised."""
        try:
            return super()._deserialize(value, attr, data)
        except ValidationError as ex:
            if 'deserialization_schema_selector' in ex.messages[0]:
                ex.messages[0] = 'Cannot find a valid schema among the choices'
            raise

    def _serialize(self, value, key, obj):
        """Override _serialize for customizing the Exception raised."""
        try:
            return super()._serialize(value, key, obj)
        except TypeError as ex:
            if 'serialization_schema_selector' in str(ex):
                raise ValidationError('Data from an invalid schema')
            raise

    def _validate_model(self, value, attr, data):
        """Helper for minimal validation of fields.BasePolyField."""
        if not self.many:
            values = [value]
        else:
            values = value

        for v in values:
            schema = self.serialization_schema_selector(v, data)
            if not schema:
                raise _not_expected_type(
                    value, self.__class__, fields=[self], field_names=attr,
                    data=data)

        return value


class TryFrom(BasePolyField):
    """Polymorphic field that returns the first candidate schema that matches.

    Polymorphic field that accepts a list of candidate schemas, and iterates
    through them, returning the first Schema that matches the data. Note that
    the list of choices is traversed in order, and an attempt to match the
    data is performed until a valid Schema is found, which might have
    performance implications.

    Examples:
        class PetOwnerSchema(BaseSchema):
            pet = TryFrom([CatSchema, DogSchema])

    Args:
        choices (list[class]): list of BaseSchema classes that are iterated in
            order for for performing disambiguation.
        many (bool): whether the field is a collection of objects.
        metadata (dict): the same keyword arguments that ``PolyField`` receives.
    """

    def to_dict_selector(self, choices, base_object, *_):
        # pylint: disable=arguments-differ
        if getattr(base_object, 'schema'):
            if base_object.schema.__class__ in choices:
                return base_object.schema

        return None

    def from_dict_selector(self, choices, base_dict, *_):
        # pylint: disable=arguments-differ
        for schema_cls in choices:
            try:
                schema = schema_cls(strict=True)
                schema.load(base_dict)

                return schema_cls()
            except ValidationError:
                pass
        return None


class ByAttribute(BasePolyField):
    """Polymorphic field that disambiguates based on an attribute's existence.

    Polymorphic field that accepts a dictionary of (``'attribute': schema``)
    entries, and checks for the existence of ``attribute`` in the data for
    disambiguating.

    Examples:
        class PetOwnerSchema(BaseSchema):
            pet = ByAttribute({'fur_density': CatSchema,
                               'barking_power': DogSchema)}

    Args:
        choices (dict[string: class]): dictionary with attribute names as
            keys, and BaseSchema classes as values.
        many (bool): whether the field is a collection of objects.
        metadata (dict): the same keyword arguments that ``PolyField`` receives.
    """

    def to_dict_selector(self, choices, base_object, *_):
        # pylint: disable=arguments-differ
        if getattr(base_object, 'schema'):
            if base_object.schema.__class__ in choices.values():
                return base_object.schema

        return None

    def from_dict_selector(self, choices, base_dict, *_):
        # pylint: disable=arguments-differ
        for attribute, schema_cls in choices.items():
            if attribute in base_dict:
                return schema_cls()

        return None


class ByType(_fields.Field):
    """Polymorphic field that disambiguates based on an attribute's type.

    Polymorphic field that accepts a list of ``Fields``, and checks that the
    data belongs to any of those types. Note this Field does not inherit from
    ``BasePolyField``, as it operates directly on ``Fields`` instead of
    operating in ``Schemas``.

    Examples:
        class PetOwnerSchema(BaseSchema):
            contact_method = ByType([fields.Email(), fields.Url()])

    Args:
        choices (list[Field]): list of accepted `Fields` instances.
        *args (tuple): args for Field.
        **kwargs (dict): kwargs for Field.
    """

    default_error_messages = {
        'invalid': 'Value {value} does not fit any of the types {types}.'
    }

    def __init__(self, choices, *args, **kwargs):
        self.choices = choices
        super().__init__(*args, **kwargs)

    def _serialize(self, value, attr, obj):
        for field in self.choices:
            try:
                return field._serialize(value, attr, obj)
            except ValidationError:
                pass

        self.fail('invalid', value=value, types=self.choices)

    def _deserialize(self, value, attr, data):
        for field in self.choices:
            try:
                return field._deserialize(value, attr, data)
            except ValidationError:
                pass

        self.fail('invalid', value=value, types=self.choices)

    def _validate_model(self, value, attr, data):
        for field in self.choices:
            if hasattr(field, '_validate_model'):
                try:
                    return getattr(field, '_validate_model')(value, attr, data)
                except ValidationError:
                    pass

        raise _not_expected_type(
            value, [field.__class__ for field in self.choices],
            fields=[self], field_names=attr, data=data)


class Complex(_fields.Field):
    """Field for complex numbers.

    Field for parsing complex numbers:
    * deserializes to Python's `complex`.
    * serializes to a tuple of 2 decimals `(float, imaginary)`
    """

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

    def _validate_model(self, value, attr, data):
        if not isinstance(value, complex):
            raise _not_expected_type(
                value, complex, fields=[self], field_names=attr, data=data)

        return value


class String(_fields.String):
    # pylint: disable=missing-docstring
    __doc__ = _fields.String.__doc__

    def _validate_model(self, value, attr, data):
        if not isinstance(value, str):
            raise _not_expected_type(
                value, str, fields=[self], field_names=attr, data=data)

        return value


class Date(_fields.Date):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Date.__doc__

    def _validate_model(self, value, attr, data):
        if not isinstance(value, date):
            raise _not_expected_type(
                value, date, fields=[self], field_names=attr, data=data)

        return value


class DateTime(_fields.DateTime):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Date.__doc__

    def _validate_model(self, value, attr, data):
        if not isinstance(value, date):
            raise _not_expected_type(
                value, datetime, fields=[self], field_names=attr, data=data)

        return value


class Email(String, _fields.Email):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Email.__doc__


class Url(String, _fields.Email):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Email.__doc__


class Nested(_fields.Nested):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Nested.__doc__

    def _validate_model(self, value, attr, data):
        if self.many and not is_collection(value):
            self.fail('type', input=value, type=value.__class__.__name__)

        if not self.many:
            values = [value]
        else:
            values = value

        for v in values:
            if not isinstance(v, self.schema.model_cls):
                raise _not_expected_type(
                    value, self.schema.model_cls, fields=[self], field_names=attr,
                    data=data)

        return value


class List(_fields.List):
    # pylint: disable=missing-docstring
    __doc__ = _fields.List.__doc__

    def _validate_model(self, value, *_):
        errors = []
        if not is_collection(value):
            raise _not_expected_type(value, 'iterable')

        for idx, v in enumerate(value):
            if hasattr(self.container, '_validate_model'):
                try:
                    getattr(self.container, '_validate_model')(v, idx, value)
                except ValidationError as err:
                    errors.append(err.messages)

        if errors:
            raise ValidationError(errors)

        return value


class Number(_fields.Number):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Number.__doc__

    def _validate_model(self, value, attr, data):
        if not isinstance(value, (int, float)):
            raise _not_expected_type(
                value, (int, float), fields=[self], field_names=attr, data=data)

        return value


class Integer(_fields.Integer):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Integer.__doc__

    def _validate_model(self, value, attr, data):
        if not isinstance(value, int):
            raise _not_expected_type(
                value, int, fields=[self], field_names=attr, data=data)

        return value


class Float(_fields.Float):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Float.__doc__

    def _validate_model(self, value, attr, data):
        if not isinstance(value, float):
            raise _not_expected_type(
                value, float, fields=[self], field_names=attr, data=data)

        return value


class Boolean(_fields.Boolean):
    # pylint: disable=missing-docstring
    __doc__ = _fields.Boolean.__doc__

    def _validate_model(self, value, attr, data):
        if not isinstance(value, bool):
            raise _not_expected_type(
                value, bool, fields=[self], field_names=attr, data=data)

        return value


def _not_expected_type(value, type_, **kwargs):
    message = 'Value {} not of expected type {}'.format(value, type_)
    return ValidationError(message, **kwargs)
