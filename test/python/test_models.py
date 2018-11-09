# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Models tests."""

from marshmallow import fields, ValidationError

from qiskit.validation.base import BaseModel, BaseSchema, bind_schema
from qiskit.validation.fields import TryFrom
from .common import QiskitTestCase


# Example classes for testing instantiation.

class PersonSchema(BaseSchema):
    """Example Person schema."""
    name = fields.String(required=True)


class BookSchema(BaseSchema):
    """Example Book schema."""
    title = fields.String(required=True)
    date = fields.Date()
    author = fields.Nested(PersonSchema, required=True)


@bind_schema(PersonSchema)
class Person(BaseModel):
    """Example Person model."""
    pass


@bind_schema(BookSchema)
class Book(BaseModel):
    """Example Book model."""
    pass


class TestModels(QiskitTestCase):
    """Tests for models."""

    def test_instantiate(self):
        """Test model instantiation."""
        person = Person(name='Foo')
        self.assertEqual(person.name, 'Foo')

    def test_instantiate_required(self):
        """Test model instantiation without required fields."""
        with self.assertRaises(ValidationError):
            _ = Person()

    def test_instantiate_additional(self):
        """Test model instantiation with additional fields."""
        person = Person(name='Foo', other='bar')
        self.assertEqual(person.other, 'bar')

    def test_instantiate_from_dict(self):
        """Test model instantiation from dictionary, with additional fields."""
        person = Person.from_dict({'name': 'Foo', 'other': 'bar'})
        self.assertEqual(person.name, 'Foo')
        self.assertEqual(person.other, 'bar')

    def test_instantiate_nested(self):
        """Test model instantiation with nested fields."""
        book = Book(title='A Book', author=Person(name='Foo', other='bar'))
        self.assertEqual(book.title, 'A Book')
        self.assertEqual(book.author.name, 'Foo')
        self.assertEqual(book.author.other, 'bar')
        with self.assertRaises(AttributeError):
            _ = book.date

    def test_instantiate_nested_from_dict(self):
        """Test model instantiation from dictionary, with nested fields."""
        book = Book.from_dict({'title': 'A Book',
                               'author': {'name': 'Foo', 'other': 'bar'}})
        self.assertEqual(book.title, 'A Book')
        self.assertEqual(book.author.name, 'Foo')
        self.assertEqual(book.author.other, 'bar')
        with self.assertRaises(AttributeError):
            _ = book.date

    def test_serialize(self):
        """Test model serialization to dict."""
        person = Person(name='Foo', other='bar')
        self.assertEqual(person.to_dict(),
                         {'name': 'Foo', 'other': 'bar'})

    def test_serialize_nested(self):
        """Test model serialization to dict, with nested fields."""
        book = Book(title='A Book', author=Person(name='Foo', other='bar'))
        self.assertEqual(book.to_dict(),
                         {'title': 'A Book',
                          'author': {'name': 'Foo', 'other': 'bar'}})


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


# Example classes for testing fields.

class DogSchema(BaseSchema):
    """Example Dog schema."""
    barking_power = fields.Integer(required=True)


class CatSchema(BaseSchema):
    """Example Cat schema."""
    fur_density = fields.Float(required=True)


class PetOwnerSchema(BaseSchema):
    """Example PetOwner schema."""
    auto_pets = TryFrom([CatSchema, DogSchema], many=True)


@bind_schema(DogSchema)
class Dog(BaseModel):
    """Example Dog model."""
    pass


@bind_schema(CatSchema)
class Cat(BaseModel):
    """Example Cat model."""
    pass


@bind_schema(PetOwnerSchema)
class PetOwner(BaseModel):
    """Example PetOwner model."""
    pass


class TestFields(QiskitTestCase):
    """Tests for Fields."""

    def test_try_from_field_instantiate(self):
        """Test the TryFrom field, instantiation."""
        pet_owner = PetOwner(auto_pets=[Cat(fur_density=1.5),
                                        Dog(barking_power=100)])
        self.assertIsInstance(pet_owner.auto_pets[0], Cat)
        self.assertIsInstance(pet_owner.auto_pets[1], Dog)
        self.assertEqual(pet_owner.auto_pets[0].fur_density, 1.5)

    def test_try_from_field_instantiate_from_dict(self):
        """Test the TryFrom field, instantiation from dict."""
        pet_owner = PetOwner.from_dict({'auto_pets': [{'fur_density': 1.5},
                                                      {'barking_power': 100}]})
        self.assertIsInstance(pet_owner.auto_pets[0], Cat)
        self.assertIsInstance(pet_owner.auto_pets[1], Dog)
        self.assertEqual(pet_owner.auto_pets[0].fur_density, 1.5)

    def test_try_from_field_invalid(self):
        """Test the TryFrom field, with invalid kind of object."""
        with self.assertRaises(ValidationError) as context_manager:
            _ = PetOwner(auto_pets=[Cat(fur_density=1.5),
                                    Person(name='John Doe')])
        self.assertIn('auto_pet', str(context_manager.exception))
