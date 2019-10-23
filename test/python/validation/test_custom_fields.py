# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test terra custom fields."""

import numpy
import sympy

from qiskit.circuit import Parameter

from qiskit.test import QiskitTestCase
from qiskit.validation import fields
from qiskit.validation.base import BaseModel, BaseSchema, bind_schema
from qiskit.validation.exceptions import ModelValidationError


class DogSchema(BaseSchema):
    """Example Dog schema."""
    barking = fields.InstructionParameter(required=True)


@bind_schema(DogSchema)
class Dog(BaseModel):
    """Example Dog model."""
    pass


class TestFields(QiskitTestCase):
    """Tests for Fields."""

    def test_parameter_field_float(self):
        """Test InstructionParameter with types equivalent to float."""
        dog = Dog(barking=0.1)
        dog_numpy = Dog(barking=numpy.dtype('float').type(0.1))

        self.assertEqual(dog_numpy.barking,
                         numpy.dtype('float').type(0.1))
        self.assertEqual(dog.to_dict(), dog_numpy.to_dict())

    def test_parameter_field_complex(self):
        """Test InstructionParameter with types equivalent to complex."""
        dog = Dog(barking=(0.1+0.2j))
        dog_numpy = Dog(barking=numpy.dtype('complex').type(0.1+0.2j))

        self.assertEqual(dog_numpy.barking,
                         numpy.dtype('complex').type(0.1+0.2j))
        self.assertEqual(dog.to_dict(), dog_numpy.to_dict())
        self.assertEqual(dog.to_dict()['barking'], [0.1, 0.2])

    def test_parameter_field_int(self):
        """Test InstructionParameter with types equivalent to int."""
        dog = Dog(barking=1)
        dog_numpy = Dog(barking=numpy.dtype('int').type(1))

        self.assertEqual(dog_numpy.barking,
                         numpy.dtype('int').type(1))
        self.assertEqual(dog.to_dict(), dog_numpy.to_dict())

    def test_parameter_field_str(self):
        """Test InstructionParameter with types equivalent to str."""
        dog = Dog(barking='woof')

        self.assertEqual(dog.barking, 'woof')
        self.assertEqual(dog.to_dict(), {'barking': 'woof'})

    def test_parameter_field_container(self):
        """Test InstructionParameter with container types."""
        dog = Dog(barking=[0.1, 2.3])
        dog_numpy = Dog(barking=numpy.array([0.1, 2.3]))
        dog_complex = Dog(barking=complex(0.1, 2.3))

        self.assertEqual(dog.to_dict(), dog_numpy.to_dict())
        self.assertEqual(dog.to_dict(), dog_complex.to_dict())

    def test_parameter_field_container_nested(self):
        """Test InstructionParameter with nested container types."""
        dog = Dog(barking=[[1, 2]])
        dog_numpy = Dog(barking=numpy.array([[1, 2]]))

        self.assertEqual(dog.to_dict(), dog_numpy.to_dict())

    def test_parameter_field_from_dict(self):
        """Test InstructionParameter from_dict."""
        dog = Dog.from_dict({'barking': [0.1, 2]})
        self.assertEqual(dog.barking, [0.1, 2])

    def test_parameter_field_invalid(self):
        """Test InstructionParameter invalid values."""
        with self.assertRaises(ModelValidationError) as context_manager:
            _ = Dog(barking={})
        self.assertIn('barking', str(context_manager.exception))

        with self.assertRaises(ModelValidationError) as context_manager:
            # Invalid types inside containers are also not allowed.
            _ = Dog(barking=[{}])
        self.assertIn('barking', str(context_manager.exception))

    def test_parameter_field_from_dict_invalid(self):
        """Test InstructionParameter from_dict."""
        with self.assertRaises(ModelValidationError) as context_manager:
            _ = Dog.from_dict({'barking': {}})
        self.assertIn('barking', str(context_manager.exception))

        with self.assertRaises(ModelValidationError) as context_manager:
            # Invalid types inside containers are also not allowed.
            _ = Dog.from_dict({'barking': [{}]})
        self.assertIn('barking', str(context_manager.exception))

        with self.assertRaises(ModelValidationError) as context_manager:
            # Non-basic types cannot be used in from_dict, even if they are
            # accepted in the constructor. This is by design, as the serialized
            # form should only contain Python/json-schema basic types.
            _ = Dog.from_dict({'barking': sympy.Symbol('woof')})
        self.assertIn('barking', str(context_manager.exception))

    def test_parameter_expression_fully_bound(self):
        """Test ParameterExpressions valid after fully bound."""
        x = Parameter('x')
        y = Parameter('y')

        expr = x
        bound_expr = expr.bind({x: 2})

        dog_expr = Dog(barking=bound_expr)
        dog_float = Dog(barking=2)
        self.assertEqual(dog_expr.barking, bound_expr)
        self.assertEqual(dog_expr.to_dict(), dog_float.to_dict())

        expr = x + y
        bound_expr = expr.bind({x: 2, y: 3})

        dog_expr = Dog(barking=bound_expr)
        dog_float = Dog(barking=5)
        self.assertEqual(dog_expr.barking, bound_expr)
        self.assertEqual(dog_expr.to_dict(), dog_float.to_dict())

    def test_parameter_expression_partially_bound(self):
        """Test ParameterExpressions invalid if partially bound."""
        x = Parameter('x')
        y = Parameter('y')

        with self.assertRaises(ModelValidationError) as context_manager:
            _ = Dog(barking=x).to_dict()
        self.assertIn('barking', str(context_manager.exception))

        expr = x + y

        with self.assertRaises(ModelValidationError) as context_manager:
            _ = Dog(barking=expr).to_dict()
        self.assertIn('barking', str(context_manager.exception))

        partially_bound_expr = expr.bind({x: 2})

        with self.assertRaises(ModelValidationError) as context_manager:
            _ = Dog(barking=partially_bound_expr).to_dict()
        self.assertIn('barking', str(context_manager.exception))
