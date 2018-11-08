# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Building blocks for Qiskit validated classes.

The module contains base classes for validated classes and schemas and the
``bind_schema`` decorator to orchestrate them."""

from functools import wraps
from types import SimpleNamespace

from marshmallow import ValidationError
from marshmallow import Schema, post_dump, post_load


class ModelSchema(Schema):
    """Provide deserialization into class instances instead of dicts.

    Conveniently for the Qiskit common case, this class also loads and dumps
    unknown attributes not defined in the schema.

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

        From https://github.com/marshmallow-code/marshmallow/pull/595.
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


class _BindSchema:
    """Aux class to implement the parametrized decorator ``bind_schema``.

    TODO:
        - Raise if trying to bind a schema more than once.
    """

    def __init__(self, schema):
        """Get the schema for the decorated model."""
        self._schema = schema

    def __call__(self, model_cls):
        """Augment the model class with the validation API.

        See the docs for ``bind_schema`` for further information.
        """
        self._schema.model_cls = model_cls
        model_cls.schema = self._schema()
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


class BaseModel(SimpleNamespace):
    """Root class for validated Qiskit classes."""
    pass


def bind_schema(schema):
    """By decorating a class, it adds schema validation to its instances.

    Instances of the decorated class are automatically validated after
    instantiation and they are augmented to allow further validations with the
    private method ``_validate()``.

    The decorator also adds the class attribute ``schema`` with the schema used
    for validation.

    To ease serialization/deserialization to/from simple Python objects,
    classes are provided with ``to_dict`` and ``from_dict`` instance and class
    methods respectively.
    """
    return _BindSchema(schema)
