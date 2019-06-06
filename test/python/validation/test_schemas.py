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
