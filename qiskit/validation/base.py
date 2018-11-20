# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

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

from functools import partial, wraps
from types import SimpleNamespace

from marshmallow import ValidationError
from marshmallow import Schema, post_dump, post_load, fields
from marshmallow.utils import is_collection

from .fields import BasePolyField, ByType


class BaseSchema(Schema):
    """Base class for Schemas for validated Qiskit classes.

    Provides convenience functionality for the Qiskit common use case:

    * deserialization into class instances instead of dicts.
    * handling of unknown attributes not defined in the schema.

    Attributes:
         model_cls (type): class used to instantiate the instance. The
         constructor is passed all named parameters from deserialization.
    """

    model_cls = SimpleNamespace

    @post_dump(pass_original=True, pass_many=True)
    def dump_additional_data(self, valid_data, many, original_data):
        """Include unknown fields after dumping.

        Unknown fields are added with no processing at all.

        Args:
            valid_data (dict or list): data collected and returned by ``dump()``.
            many (bool): if True, data and original_data are a list.
            original_data (object or list): object passed to ``dump()`` in the
                first place.

        Returns:
            dict: the same ``valid_data`` extended with the unknown attributes.

        Inspired by https://github.com/marshmallow-code/marshmallow/pull/595.
        """
        if many:
            for i, _ in enumerate(valid_data):
                additional_keys = set(original_data[i].__dict__) - set(valid_data[i])
                for key in additional_keys:
                    valid_data[i][key] = getattr(original_data[i], key)
        else:
            additional_keys = set(original_data.__dict__) - set(valid_data)
            for key in additional_keys:
                valid_data[key] = getattr(original_data, key)

        return valid_data

    @post_load(pass_original=True, pass_many=True)
    def load_additional_data(self, valid_data, many, original_data):
        """Include unknown fields after load.

        Unknown fields are added with no processing at all.

        Args:
            valid_data (dict or list): validated data returned by ``load()``.
            many (bool): if True, data and original_data are a list.
            original_data (dict or list): data passed to ``load()`` in the
                first place.

        Returns:
            dict: the same ``valid_data`` extended with the unknown attributes.

        Inspired by https://github.com/marshmallow-code/marshmallow/pull/595.
        """
        if many:
            for i, _ in enumerate(valid_data):
                additional_keys = set(original_data[i]) - set(valid_data[i])
                for key in additional_keys:
                    valid_data[i][key] = original_data[i][key]
        else:
            additional_keys = set(original_data) - set(valid_data)
            for key in additional_keys:
                valid_data[key] = original_data[key]

        return valid_data

    @post_load
    def make_model(self, data):
        """Make ``load`` return a ``model_cls`` instance instead of a dict."""
        return self.model_cls(**data)


class _SchemaBinder:
    """Helper class for the parametrized decorator ``bind_schema``."""

    def __init__(self, schema_cls):
        """Get the schema for the decorated model."""
        self._schema_cls = schema_cls

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

        # Set a reference to the Model in the Schema, and viceversa.
        self._schema_cls.model_cls = model_cls
        model_cls.schema = self._schema_cls()

        # Append the methods to the Model class.
        model_cls.to_dict = self._to_dict
        model_cls.from_dict = classmethod(self._from_dict)
        model_cls._validate = self._validate
        model_cls.__init__ = self._validate_after_init(model_cls.__init__)

        # Add a Schema that performs minimal validation to the Model.
        model_cls.shallow_schema = self._create_shallow_schema(self._schema_cls)

        return model_cls

    def _create_shallow_schema(self, schema_cls):
        """Create a Schema with minimal validation for compound types.


        This is a helper for performing the initial validation when
        instantiating the Model via **kwargs. It works on the assumption that
        **kwargs will contain:
        * for compound types (`Nested`, `BasePolyField`), it will already
          contain `BaseModels`, which should have been validated earlier
          (during _their_ instantiation), and only type checking is performed.
        * for `Number` and `String` types, both serialized and deserialized
          are equivalent, and the shallow_schema will try to serialize in
          order to perform stronger validation.
        * for the rest of fields (the ones where the serialized and deserialized
          data is different), it will contain _deserialized_ types that are
          passed through.

        The underlying idea is to be able to perform validation (in the schema)
        at only the first level of the object, and at the same time take
        advantage of validation during **kwargs instantiation as much as
        possible (mimicking `.from_dict()` in that respect).

        Returns:
            BaseSchema: a copy of the original Schema, overriding the
                ``_deserialize()`` call of its fields.
        """
        shallow_schema = schema_cls()
        for _, field in shallow_schema.fields.items():
            if isinstance(field, fields.Nested):
                field._deserialize = partial(self._overridden_nested_deserialize, field)
            elif isinstance(field, BasePolyField):
                field._deserialize = partial(self._overridden_basepolyfield_deserialize, field)
            elif not isinstance(field, (fields.Number, fields.String, ByType)):
                field._deserialize = partial(self._overridden_field_deserialize, field)
        return shallow_schema

    @staticmethod
    def _overridden_nested_deserialize(field, value, _, data):
        """Helper for minimal validation of fields.Nested."""
        if field.many and not is_collection(value):
            field.fail('type', input=value, type=value.__class__.__name__)

        if not field.many:
            values = [value]
        else:
            values = value

        for v in values:
            if not isinstance(v, field.schema.model_cls):
                raise ValidationError(
                    'Not a valid type for {}.'.format(field.__class__.__name__),
                    data=data)
        return value

    @staticmethod
    def _overridden_basepolyfield_deserialize(field, value, _, data):
        """Helper for minimal validation of fields.BasePolyField."""
        if not field.many:
            values = [value]
        else:
            values = value

        for v in values:
            schema = field.serialization_schema_selector(v, data)
            if not schema:
                raise ValidationError(
                    'Not a valid type for {}.'.format(field.__class__.__name__),
                    data=data)
        return value

    @staticmethod
    def _overridden_field_deserialize(field, value, attr, data):
        """Helper for validation of generic Field."""
        # Attempt to serialize, in order to catch validation errors.
        field._serialize(value, attr, data)

        # Propagate the original value upwards.
        return value

    @staticmethod
    def _to_dict(instance):
        """Serialize the model into a Python dict of simple types."""
        data, errors = instance.schema.dump(instance)
        if errors:
            raise ValidationError(errors)
        return data

    @staticmethod
    def _validate(instance):
        """Validate the internal representation of the instance."""
        errors = instance.schema.validate(instance.to_dict())
        if errors:
            raise ValidationError(errors)

    @staticmethod
    def _from_dict(decorated_cls, dict_):
        """Deserialize a dict of simple types into an instance of this class."""
        data, errors = decorated_cls.schema.load(dict_)
        if errors:
            raise ValidationError(errors)
        return data

    @staticmethod
    def _validate_after_init(init_method):
        """Add validation after instantiation."""

        @wraps(init_method)
        def _decorated(self, **kwargs):
            errors = self.shallow_schema.validate(kwargs)
            if errors:
                raise ValidationError(errors)

            init_method(self, **kwargs)

        return _decorated


def bind_schema(schema):
    """Class decorator for adding schema validation to its instances.

    Instances of the decorated class are automatically validated after
    instantiation and they are augmented to allow further validations with the
    private method ``_validate()``.

    The decorator also adds the class attribute ``schema`` with the schema used
    for validation, along with a class attribute ``shallow_schema`` used for
    validation during instantiation.

    To ease serialization/deserialization to/from simple Python objects,
    classes are provided with ``to_dict`` and ``from_dict`` instance and class
    methods respectively.

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

    Raises:
        ValueError: when trying to bind the same schema more than once.

    Return:
        type: the same class with validation capabilities.
    """
    return _SchemaBinder(schema)


def _base_model_from_kwargs(cls, kwargs):
    """Helper for BaseModel.__reduce__, expanding kwargs."""
    return cls(**kwargs)


class BaseModel(SimpleNamespace):
    """Base class for Models for validated Qiskit classes."""
    def __reduce__(self):
        """Custom __reduce__ for allowing pickling and unpickling.

        Customize the reduction in order to allow serialization, as the
        BaseModels need to be pickled during the use of futures by the backends.
        Instead of returning the class, a helper is used in order to pass the
        arguments as **kwargs, as it is needed by SimpleNamespace and the
        standard __reduce__ only allows passing args as a tuple.
        """
        return _base_model_from_kwargs, (self.__class__, self.__dict__)


class ObjSchema(BaseSchema):
    """Generic object schema."""
    pass


@bind_schema(ObjSchema)
class Obj(BaseModel):
    """Generic object in a Model."""
    pass
