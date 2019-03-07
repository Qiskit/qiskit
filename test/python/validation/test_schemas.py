# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test qiskit schema binding."""

from qiskit.validation.base import BaseModel, BaseSchema, bind_schema
from qiskit.test import QiskitTestCase


class TestSchemas(QiskitTestCase):
    """Tests for operations with schemas."""

    def test_double_binding(self):
        """Trying to bind a schema twice must raise."""
        class _DummySchema(BaseSchema):
            pass

        @bind_schema(_DummySchema)
        class _DummyModel(BaseModel):
            pass

        with self.assertRaises(ValueError):
            @bind_schema(_DummySchema)
            class _AnotherModel(BaseModel):
                pass

    def test_schema_reusing(self):
        """Reusing a schema is possible if subclassing."""
        class _DummySchema(BaseSchema):
            pass

        class _SchemaCopy(_DummySchema):
            pass

        @bind_schema(_DummySchema)
        class _DummyModel(BaseModel):
            pass

        try:
            @bind_schema(_SchemaCopy)
            class _AnotherModel(BaseModel):
                pass
        except ValueError:
            self.fail('`bind_schema` raised while binding.')
