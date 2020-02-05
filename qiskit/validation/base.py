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

"""Building blocks for Qiskit validated classes.

This module provides the ``BaseSchema`` and ``BaseModel`` classes as the main
building blocks for defining objects (Models) that conform to a specification
(Schema) and are validated at instantiation, along with providing facilities
for being serialized and deserialized.

Implementors are recommended to subclass the two classes, and "binding" them
together by using ``bind_schema``::

    class PersonSchema(BaseSchema):
        name = String(required=True)

    @bind_schema(PersonSchema)
    class Person(BaseModel):
        pass
"""

from functools import wraps
from types import SimpleNamespace, MethodType

from marshmallow import ValidationError
from marshmallow import Schema, post_dump, post_load
from marshmallow import fields as _fields
from marshmallow.utils import is_collection, INCLUDE

from .exceptions import ModelValidationError


class ModelTypeValidator(_fields.Field):
    """A field able to validate the correct type of a value."""

    valid_types = (object, )

    def _expected_types(self):
        return self.valid_types

    def check_type(self, value, attr, data, **_):
        """Validates a value against the correct type of the field.

        It calls ``_expected_types`` to get a list of valid types.

        Subclasses can do one of the following:

        1. Override the ``valid_types`` property with a tuple with the expected
           types for this field.

        2. Override the ``_expected_types`` method to return a tuple of
           expected types for the field.

        3. Change ``check_type`` completely to customize validation.

        Note:
            This method or the overrides must return the ``value`` parameter
            untouched.
        """
        expected_types = self._expected_types()
        if not isinstance(value, expected_types):
            raise self._not_expected_type(
                value, expected_types, fields=[self], field_names=attr, data=data)
        return value

    @staticmethod
    def _not_expected_type(value, type_, **kwargs):
        if is_collection(type_) and len(type_) == 1:
            type_ = type_[0]

        if is_collection(type_):
            body = 'is none of the expected types {}'.format(type_)
        else:
            body = 'is not the expected type {}'.format(type_)

        message = 'Value \'{}\' {}: {}'.format(value, type(value), body)
        return ValidationError(message, **kwargs)

    def make_error_serialize(self, key, **kwargs):
        """Helper method to return a ValidationError from _serialize.

        This method wraps the result of ``make_error()``, adding contextual
        information in order to provide more informative information to users.

        Args:
            key (str): error key index.
            **kwargs: additional arguments to ``make_error()``.

        Returns:
            ValidationError: an exception with the field name.
        """
        bare_error = self.make_error(key, **kwargs)
        return ValidationError({self.name: bare_error.messages},
                               field_name=self.name)


class BaseSchema(Schema):
    """Base class for Schemas for validated Qiskit classes.

    Provides convenience functionality for the Qiskit common use case:

    * deserialization into class instances instead of dicts.
    * handling of unknown attributes not defined in the schema.

    Attributes:
         model_cls (type): class used to instantiate the instance. The
         constructor is passed all named parameters from deserialization.
    """

    class Meta:
        """Add extra fields to the schema."""
        unknown = INCLUDE

    model_cls = SimpleNamespace

    @post_dump(pass_original=True, pass_many=True)
    def dump_additional_data(self, valid_data, original_data, **kwargs):
        """Include unknown fields after dumping.

        Unknown fields are added with no processing at all.

        Args:
            valid_data (dict or list): data collected and returned by ``dump()``.
            original_data (object or list): object passed to ``dump()`` in the
                first place.
            **kwargs: extra arguments from the decorators.

        Returns:
            dict: the same ``valid_data`` extended with the unknown attributes.

        Inspired by https://github.com/marshmallow-code/marshmallow/pull/595.
        """
        if kwargs.get('many'):
            for i, _ in enumerate(valid_data):
                additional_keys = set(original_data[i].__dict__) - set(valid_data[i])
                for key in additional_keys:
                    if key.startswith('_'):
                        continue
                    valid_data[i][key] = getattr(original_data[i], key)
        else:
            additional_keys = set(original_data.__dict__) - set(valid_data)
            for key in additional_keys:
                if key.startswith('_'):
                    continue
                valid_data[key] = getattr(original_data, key)

        return valid_data

    @post_load
    def make_model(self, data, **_):
        """Make ``load`` return a ``model_cls`` instance instead of a dict."""
        return self.model_cls(**data)


class _SchemaBinder:
    """Helper class for the parametrized decorator ``bind_schema``."""

    def __init__(self, schema_cls, **kwargs):
        """Get the schema for the decorated model."""
        self._schema_cls = schema_cls
        self._kwargs = kwargs

    def __call__(self, model_cls):
        """Augment the model class with the validation API.

        See the docs for ``bind_schema`` for further information.
        """
        # Check for double binding of schemas.
        if self._schema_cls.__dict__.get('model_cls', None) is not None:
            raise ValueError(
                'The schema {} can not be bound twice. It is already bound to '
                '{}. If you want to reuse the schema, use '
                'subclassing'.format(self._schema_cls, self._schema_cls.model_cls))

        # Set a reference to the Model in the Schema, and vice versa.
        self._schema_cls.model_cls = model_cls
        model_cls.schema = self._schema_cls(**self._kwargs)

        # Append the methods to the Model class.
        model_cls.__init__ = self._validate_after_init(model_cls.__init__)

        # Add a Schema that performs minimal validation to the Model.
        model_cls.shallow_schema = self._create_validation_schema(self._schema_cls)

        return model_cls

    @staticmethod
    def _create_validation_schema(schema_cls, **kwargs):
        """Create a patched Schema for validating models.

        Model validation is not part of Marshmallow. Schemas have a ``validate``
        method but this delegates execution on ``load``. Similarly, ``load``
        will call ``_deserialize`` on every field in the schema.

        This function patches the ``_deserialize`` instance method of each
        field to make it call a custom defined method ``check_type``
        provided by Qiskit in the different fields at
        ``qiskit.validation.fields``.

        Returns:
            BaseSchema: a copy of the original Schema, overriding the
                ``_deserialize()`` call of its fields.
        """
        validation_schema = schema_cls(**kwargs)
        for _, field in validation_schema.fields.items():
            if isinstance(field, ModelTypeValidator):
                validate_function = field.__class__.check_type
                field._deserialize = MethodType(validate_function, field)

        return validation_schema

    @staticmethod
    def _validate_after_init(init_method):
        """Add validation during instantiation.

        The validation is performed depending on the ``validate`` parameter
        passed to the ``init_method``. If ``False``, the validation will not be
        performed.
        """
        @wraps(init_method)
        def _decorated(self, **kwargs):
            # Extract the 'validate' parameter.
            do_validation = kwargs.pop('validate', True)
            if do_validation:
                try:
                    _ = self.shallow_schema._do_load(kwargs,
                                                     postprocess=False)
                except ValidationError as ex:
                    raise ModelValidationError(
                        ex.messages, ex.field_name, ex.data, ex.valid_data, **ex.kwargs) from None

            # Set the 'validate' parameter to False, assuming that if a
            # subclass has been validated, it superclasses will also be valid.
            return init_method(self, **kwargs, validate=False)

        return _decorated


def bind_schema(schema, **kwargs):
    """Class decorator for adding schema validation to its instances.

    The decorator acts on the model class by adding:
    * a class attribute ``schema`` with the schema used for validation
    * a class attribute ``shallow_schema`` used for validation during
      instantiation.

    The same schema cannot be bound more than once. If you need to reuse a
    schema for a different class, create a new schema subclassing the one you
    want to reuse and leave the new empty::

        class MySchema(BaseSchema):
            title = String()

        class AnotherSchema(MySchema):
            pass

        @bind_schema(MySchema):
        class MyModel(BaseModel):
            pass

        @bind_schema(AnotherSchema):
        class AnotherModel(BaseModel):
            pass

    Note:
        By default, models decorated with this decorator are validated during
        instantiation. If ``validate=False`` is passed to the constructor, this
        validation will not be performed.

    Args:
        schema (class): the schema class used for validation.
        **kwargs: Additional attributes for the ``marshmallow.Schema``
            initializer.

    Raises:
        ValueError: when trying to bind the same schema more than once.

    Return:
        type: the same class with validation capabilities.
    """
    return _SchemaBinder(schema, **kwargs)


def _base_model_from_kwargs(cls, kwargs):
    """Helper for BaseModel.__reduce__, expanding kwargs."""
    return cls(**kwargs)


class BaseModel(SimpleNamespace):
    """Base class for Models for validated Qiskit classes."""

    def __init__(self, validate=True, **kwargs):
        """BaseModel initializer.

        Note:
            The ``validate`` argument is used for controlling the behavior of
            the schema binding, and will not be present on the created object.
        """
        # pylint: disable=unused-argument
        super().__init__(**kwargs)

    def __reduce__(self):
        """Custom __reduce__ for allowing pickling and unpickling.

        Customize the reduction in order to allow serialization, as the
        BaseModels need to be pickled during the use of futures by the backends.
        Instead of returning the class, a helper is used in order to pass the
        arguments as **kwargs, as it is needed by SimpleNamespace and the
        standard __reduce__ only allows passing args as a tuple.
        """
        return _base_model_from_kwargs, (self.__class__, self.__dict__)

    def __contains__(self, item):
        """Custom implementation of membership test.

        Implement the ``__contains__`` method for catering to the common case
        of finding out if a model contains a certain key (``key in model``).
        """
        return item in self.__dict__

    def to_dict(self):
        """Serialize the model into a Python dict of simple types.

        Note that this method requires that the model is bound with
        ``@bind_schema``.
        """
        try:
            data = self.schema.dump(self)
        except ValidationError as ex:
            raise ModelValidationError(
                ex.messages, ex.field_name, ex.data, ex.valid_data, **ex.kwargs) from None

        return data

    @classmethod
    def from_dict(cls, dict_):
        """Deserialize a dict of simple types into an instance of this class.

        Note that this method requires that the model is bound with
        ``@bind_schema``.
        """
        try:
            data = cls.schema.load(dict_)
        except ValidationError as ex:
            raise ModelValidationError(
                ex.messages, ex.field_name, ex.data, ex.valid_data, **ex.kwargs) from None

        return data


class ObjSchema(BaseSchema):
    """Generic object schema."""
    pass


@bind_schema(ObjSchema)
class Obj(BaseModel):
    """Generic object in a Model."""
    pass
