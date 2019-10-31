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

"""Polymorphic fields that represent one of several schemas or types."""

from collections.abc import Iterable
from functools import partial

from marshmallow.exceptions import ValidationError
from marshmallow.utils import is_collection
from marshmallow_polyfield import PolyField

from qiskit.validation import ModelTypeValidator


class BasePolyField(PolyField, ModelTypeValidator):
    """Base class for polymorphic fields.

    Defines a Field that can contain data fitting different ``BaseSchema``.
    Deciding the type is performed by the ``to_dict_selector()`` and
    ``from_dict_selector()`` functions, that act on ``choices``.

    Subclasses are recommended to customize the ``to_dict_selector()`` and
    ``from_dict_selector()``, adding the necessary logic for inspecting
    ``choices`` and the data, and returning one of the Schemas.

     Args:
        choices (dict or iterable): iterable or dict containing the schema
        instances and the information needed for performing disambiguation.
        many (bool): whether the field is a collection of objects.
        metadata (dict): the same keyword arguments that ``PolyField`` receives.
    """

    def __init__(self, choices, many=False, **metadata):

        if isinstance(choices, dict):
            self._choices = choices.values()
        elif isinstance(choices, (list, tuple)):
            self._choices = list(choices)
        else:
            raise ValueError(
                '`choices` parameter must be a dict, a list or a tuple')

        to_dict_selector = partial(self.to_dict_selector, choices)
        from_dict_selector = partial(self.from_dict_selector, choices)

        super().__init__(to_dict_selector, from_dict_selector, many=many, **metadata)

    def to_dict_selector(self, choices, *args, **_):
        """Return an schema in ``choices`` for serialization."""
        raise NotImplementedError

    def from_dict_selector(self, choices, *args, **_):
        """Return an schema in ``choices`` for deserialization."""
        raise NotImplementedError

    def _deserialize(self, value, attr, data, **kwargs):
        """Override ``_deserialize`` for customizing the exception raised."""
        # pylint: disable=arguments-differ
        try:
            return super()._deserialize(value, attr, data, **kwargs)
        except ValidationError as ex:
            if 'deserialization_schema_selector' in ex.messages[0]:
                ex.messages[0] = 'Cannot find a valid schema among the choices'
            raise

    def _serialize(self, value, key, obj, **kwargs):
        """Override ``_serialize`` for customizing the exception raised."""
        try:
            return super()._serialize(value, key, obj, **kwargs)
        except TypeError as ex:
            if 'serialization_schema_selector' in str(ex):
                raise ValidationError('Data from an invalid schema')
            raise

    def _expected_types(self):
        return tuple(schema.model_cls for schema in self._choices)

    def check_type(self, value, attr, data, **kwargs):
        """Check if the type of the value is one of the possible choices.

        Possible choices are the model classes bound to the possible schemas.
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
                schema = schema_cls()
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


class ByType(ModelTypeValidator):
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
        *args: args for Field.
        **kwargs: kwargs for Field.
    """

    default_error_messages = {
        'invalid': 'Value {value} does not fit any of the types {types}.'
    }

    def __init__(self, choices, *args, **kwargs):
        self.choices = choices
        super().__init__(*args, **kwargs)

    def _serialize(self, value, attr, obj, **kwargs):
        for field in self.choices:
            try:
                return field._serialize(value, attr, obj, **kwargs)
            except (ValidationError, ValueError):
                pass

        raise self.make_error_serialize('invalid', value=value, types=self.choices)

    def _deserialize(self, value, attr, data, **kwargs):
        for field in self.choices:
            try:
                return field._deserialize(value, attr, data, **kwargs)
            except (ValidationError, ValueError):
                pass

        raise self.make_error('invalid', value=value, types=self.choices)

    def check_type(self, value, attr, data, **kwargs):
        """Check if at least one of the possible choices validates the value.

        Possible choices are assumed to be ``ModelTypeValidator`` fields.
        """
        for field in self.choices:
            if isinstance(field, ModelTypeValidator):
                try:
                    return field.check_type(value, attr, data, **kwargs)
                except ValidationError:
                    pass

        raise self._not_expected_type(
            value, [field.__class__ for field in self.choices],
            fields=[self], field_names=attr, data=data)
