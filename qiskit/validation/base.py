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

from functools import wraps
from types import SimpleNamespace

from marshmallow import ValidationError
from marshmallow import Schema, post_dump, post_load


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

    @post_dump(pass_original=True)
    def dump_additional_data(self, valid_data, original_data):
        """Include unknown fields after dumping.

        Unknown fields are added with no processing at all.

        Args:
            valid_data (dict): data collected and returned by ``dump()``.
            original_data (object): object passed to ``dump()`` in the first
            place.

        Returns:
            dict: the same ``valid_data`` extended with the unknown attributes.

        Inspired by https://github.com/marshmallow-code/marshmallow/pull/595.
        """
        additional_keys = set(original_data.__dict__) - set(valid_data)
        for key in additional_keys:
            valid_data[key] = getattr(original_data, key)
        return valid_data

    @post_load(pass_original=True)
    def load_additional_data(self, valid_data, original_data):
        """Include unknown fields after load.

        Unknown fields are added with no processing at all.

        Args:
            valid_data (dict): validated data returned by ``load()``.
            original_data (dict): data passed to ``load()`` in the first place.

        Returns:
            dict: the same ``valid_data`` extended with the unknown attributes.

        Inspired by https://github.com/marshmallow-code/marshmallow/pull/595.
        """

        additional_keys = set(original_data) - set(valid_data)
        for key in additional_keys:
            valid_data[key] = original_data[key]
        return valid_data

    @post_load
    def make_model(self, data):
        """Make ``load`` to return an instance of ``model_cls`` instead of a dict.
        """
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
        if self._schema_cls.__dict__.get('model_cls', None) is not None:
            raise ValueError(
                'The schema {} can not be bound twice. It is already bound to '
                '{}. If you want to reuse the schema, use '
                'subclassing'.format(self._schema_cls, self._schema_cls.model_cls))

        self._schema_cls.model_cls = model_cls
        model_cls.schema = self._schema_cls()
        model_cls._validate = self._validate
        model_cls.to_dict = self._to_dict
        model_cls.from_dict = classmethod(self._from_dict)
        model_cls.__init__ = self._validate_after_init(model_cls.__init__)
        return model_cls

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
    def _from_dict(decorated_cls, dct):
        """Deserialize a dict of simple types into an instance of this class."""
        data, errors = decorated_cls.schema.load(dct)
        if errors:
            raise ValidationError(errors)
        return data

    @staticmethod
    def _validate_after_init(init_method):
        """Add validation after instantiation."""

        @wraps(init_method)
        def _decorated(self, *args, **kwargs):
            init_method(self, *args, **kwargs)
            self._validate()

        _decorated._validating = False

        return _decorated


def bind_schema(schema):
    """Class decorator for adding schema validation to its instances.

    Instances of the decorated class are automatically validated after
    instantiation and they are augmented to allow further validations with the
    private method ``_validate()``.

    The decorator also adds the class attribute ``schema`` with the schema used
    for validation.

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


class BaseModel(SimpleNamespace):
    """Base class for Models for validated Qiskit classes."""
    pass
