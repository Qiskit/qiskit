# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Models tests."""

from datetime import datetime

from marshmallow import fields, ValidationError
from marshmallow.validate import Regexp

from qiskit.validation.base import BaseModel, BaseSchema, bind_schema, ObjSchema, Obj
from qiskit.validation.fields import TryFrom, ByAttribute, ByType
from qiskit.validation.validate import PatternProperties
from .common import QiskitTestCase


# Example classes for testing instantiation.

class PersonSchema(BaseSchema):
    """Example Person schema."""
    name = fields.String(required=True)
    birth_date = fields.Date()
    email = fields.Email()


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

        # From dict.
        person_dict = Person.from_dict({'name': 'Foo'})
        self.assertEqual(person_dict.name, 'Foo')

        self.assertEqual(person, person_dict)

    def test_instantiate_required(self):
        """Test model instantiation without required fields."""
        with self.assertRaises(ValidationError):
            _ = Person()

        # From dict.
        with self.assertRaises(ValidationError):
            _ = Person.from_dict({})

    def test_instantiate_wrong_type(self):
        """Test model instantiation with fields of the wrong type."""
        with self.assertRaises(ValidationError):
            _ = Person(name=1)

        # From dict.
        with self.assertRaises(ValidationError):
            _ = Person.from_dict({'name': 1})

    def test_instantiate_deserialized_types(self):
        """Test model instantiation with fields of deserialized type."""
        birth_date = datetime(2000, 1, 1).date()

        person = Person(name='Foo', birth_date=birth_date)
        self.assertEqual(person.birth_date, birth_date)
        with self.assertRaises(ValidationError):
            _ = Person(name='Foo', birth_date=birth_date.isoformat())

        # From dict.
        person_dict = Person.from_dict({'name': 'Foo',
                                        'birth_date': birth_date.isoformat()})
        self.assertEqual(person_dict.birth_date, birth_date)
        with self.assertRaises(ValidationError):
            _ = Person.from_dict({'name': 'Foo', 'birth_date': birth_date})

        self.assertEqual(person, person_dict)

    def test_instantiate_additional(self):
        """Test model instantiation with additional fields."""
        person = Person(name='Foo', other='bar')
        self.assertEqual(person.name, 'Foo')
        self.assertEqual(person.other, 'bar')

        # From dict.
        person_dict = Person.from_dict({'name': 'Foo', 'other': 'bar'})
        self.assertEqual(person_dict.name, 'Foo')
        self.assertEqual(person_dict.other, 'bar')

        self.assertEqual(person, person_dict)

    def test_instantiate_nested(self):
        """Test model instantiation with nested fields."""
        book = Book(title='A Book', author=Person(name='Foo', other='bar'))
        self.assertEqual(book.title, 'A Book')
        self.assertEqual(book.author.name, 'Foo')
        self.assertEqual(book.author.other, 'bar')
        with self.assertRaises(AttributeError):
            _ = book.date

        # From dict.
        book_dict = Book.from_dict({'title': 'A Book',
                                    'author': {'name': 'Foo', 'other': 'bar'}})
        self.assertEqual(book_dict.title, 'A Book')
        self.assertEqual(book_dict.author.name, 'Foo')
        self.assertEqual(book_dict.author.other, 'bar')
        with self.assertRaises(AttributeError):
            _ = book_dict.date

        self.assertEqual(book, book_dict)

    def test_instantiate_nested_wrong_type(self):
        """Test model instantiation with nested fields of the wrong type."""
        with self.assertRaises(ValidationError):
            _ = Book(title='A Book', author=Cat(fur_density=1.2))

        # From dict.
        with self.assertRaises(ValidationError):
            _ = Book.from_dict({'title': 'A Book',
                                'author': {'fur_density': '1.2'}})

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
    """Example PetOwner schema, with different polymorphic fields."""
    auto_pets = TryFrom([CatSchema, DogSchema], many=True)
    by_attribute_pets = ByAttribute({'fur_density': CatSchema,
                                     'barking_power': DogSchema}, many=True)
    by_type_contact = ByType([fields.Email(), fields.Url()])


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
        self.assertIn('auto_pets', str(context_manager.exception))

    def test_try_from_field_invalid_from_dict(self):
        """Test the TryFrom field, instantiation from dict, with invalid."""
        with self.assertRaises(ValidationError) as context_manager:
            _ = PetOwner.from_dict({'auto_pets': [{'fur_density': 1.5},
                                                  {'name': 'John Doe'}]})
        self.assertIn('auto_pets', str(context_manager.exception))

    def test_by_attribute_field_instantiate(self):
        """Test the ByAttribute field, instantiation."""
        pet_owner = PetOwner(by_attribute_pets=[Cat(fur_density=1.5),
                                                Dog(barking_power=100)])
        self.assertIsInstance(pet_owner.by_attribute_pets[0], Cat)
        self.assertIsInstance(pet_owner.by_attribute_pets[1], Dog)
        self.assertEqual(pet_owner.by_attribute_pets[0].fur_density, 1.5)

    def test_by_attribute_field_instantiate_from_dict(self):
        """Test the ByAttribute field, instantiation from dict."""
        pet_owner = PetOwner.from_dict({'by_attribute_pets': [{'fur_density': 1.5},
                                                              {'barking_power': 100}]})
        self.assertIsInstance(pet_owner.by_attribute_pets[0], Cat)
        self.assertIsInstance(pet_owner.by_attribute_pets[1], Dog)
        self.assertEqual(pet_owner.by_attribute_pets[0].fur_density, 1.5)

    def test_by_attribute_field_invalid(self):
        """Test the ByAttribute field, with invalid kind of object."""
        with self.assertRaises(ValidationError) as context_manager:
            _ = PetOwner(by_attribute_pets=[Cat(fur_density=1.5),
                                            Person(name='John Doe')])
        self.assertIn('by_attribute_pets', str(context_manager.exception))

    def test_by_attribute_field_invalid_from_dict(self):
        """Test the ByAttribute field, instantiation from dict, with invalid."""
        with self.assertRaises(ValidationError) as context_manager:
            _ = PetOwner.from_dict({'by_attribute_pets': [{'fur_density': 1.5},
                                                          {'name': 'John Doe'}]})
        self.assertIn('by_attribute_pets', str(context_manager.exception))

    def test_by_type_field_instantiate(self):
        """Test the ByType field, instantiation."""
        pet_owner = PetOwner(by_type_contact='foo@bar.com')
        self.assertEqual(pet_owner.by_type_contact, 'foo@bar.com')

    def test_by_type_field_instantiate_from_dict(self):
        """Test the ByType field, instantiation from dict."""
        pet_owner = PetOwner.from_dict({'by_type_contact': 'foo@bar.com'})
        self.assertEqual(pet_owner.by_type_contact, 'foo@bar.com')

    def test_by_type_field_invalid(self):
        """Test the ByType field, with invalid kind of object."""
        with self.assertRaises(ValidationError) as context_manager:
            _ = PetOwner(by_type_contact=123)
        self.assertIn('by_type_contact', str(context_manager.exception))

    def test_by_type_field_invalid_from_dict(self):
        """Test the ByType field, instantiation from dict, with invalid."""
        with self.assertRaises(ValidationError) as context_manager:
            _ = PetOwner.from_dict({'by_type_contact': 123})
        self.assertIn('by_type_contact', str(context_manager.exception))


class HistogramSchema(BaseSchema):
    """Example HistogramSchema schema with strict dict structure validation."""
    counts = fields.Nested(ObjSchema, validate=PatternProperties({
        Regexp('^0x([0-9A-Fa-f])+$'): fields.Integer()
    }))


@bind_schema(HistogramSchema)
class Histogram(BaseModel):
    """Example Histogram model."""
    pass


class TestValidators(QiskitTestCase):
    """Test for validators."""

    def test_patternproperties_valid(self):
        """Test the PatternProperties validator allowing fine control on keys and values."""
        counts_dict = {'0x00': 50, '0x11': 50}
        counts = Obj(**counts_dict)
        histogram = Histogram(counts=counts)
        self.assertEqual(histogram.counts, counts)

        # From dict
        histogram = Histogram.from_dict({'counts': counts_dict})
        self.assertEqual(histogram.counts, counts)

    def test_patternproperties_invalid_key(self):
        """Test the PatternProperties validator fails when invalid key"""
        invalid_key_data = {'counts': {'00': 50, '0x11': 50}}
        with self.assertRaises(ValidationError):
            _ = Histogram(**invalid_key_data)

        # From dict
        with self.assertRaises(ValidationError):
            _ = Histogram.from_dict(invalid_key_data)

    def test_patternproperties_invalid_value(self):
        """Test the PatternProperties validator fails when invalid value"""
        invalid_value_data = {'counts': {'0x00': 'so many', '0x11': 50}}
        with self.assertRaises(ValidationError):
            _ = Histogram(**invalid_value_data)

        # From dict
        with self.assertRaises(ValidationError):
            _ = Histogram.from_dict(invalid_value_data)

    def test_patternproperties_to_dict(self):
        """Test a field using the PatternProperties validator produces a correct value"""
        counts_dict = {'0x00': 50, '0x11': 50}
        counts = Obj(**counts_dict)
        histogram = Histogram(counts=counts)
        histogram_dict = histogram.to_dict()
        self.assertEqual(histogram_dict, {'counts': counts_dict})
