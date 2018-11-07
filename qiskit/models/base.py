# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Building blocks for schema validations."""

from types import SimpleNamespace

from marshmallow import Schema, post_dump, post_load


class BaseSchema(Schema):
    """Base Schema."""

    model_cls = SimpleNamespace

    class Meta:
        """Default class options."""
        unknown = 'INCLUDE'

    @post_load(pass_original=True)
    def add_original_data_load(self, valid_data, original_data):
        """Allow including unknown fields in the serialized result.

        Inspired by https://github.com/marshmallow-code/marshmallow/pull/595.
        """
        additional_keys = set(original_data) - set(valid_data)
        for key in additional_keys:
            valid_data[key] = original_data[key]
        return valid_data

    @post_dump(pass_original=True)
    def add_original_data_dump(self, valid_data, original_data):
        """Allow including unknown fields in the deserialized result.

        Inspired by https://github.com/marshmallow-code/marshmallow/pull/595.
        """
        additional_keys = set(original_data.__dict__) - set(valid_data)
        for key in additional_keys:
            valid_data[key] = getattr(original_data, key)
        return valid_data

    @post_load
    def make_model(self, data):
        """Create a new model."""
        return self.model_cls(**data, validate=False)


class MetaModel(type):
    """Meta Model."""

    def __new__(mcs, *args, **kwargs):
        cls = type.__new__(mcs, *args, **kwargs)
        if (hasattr(cls, 'schema_cls') and isinstance(cls.schema_cls, type) and
                issubclass(cls.schema_cls, BaseSchema)):
            cls.schema_cls.model_cls = cls

        return cls


class BaseModel(SimpleNamespace, metaclass=MetaModel):
    """"Base Model."""

    schema_cls = BaseSchema

    def __init__(self, validate=True, **kwargs):
        super().__init__(**kwargs)

        if validate:
            self.validate()

    def validate(self):
        """Validate."""
        # TODO: .validate() returns dict of errors, and we need full validation
        self.schema_cls().load(self.schema_cls().dump(self))

    def to_dict(self):
        """Serialize this model to a dictionary."""
        return self.schema_cls().dump(self)

    @classmethod
    def from_dict(cls, data):
        """Create a model from a dictionary."""
        return cls.schema_cls().load(data)
