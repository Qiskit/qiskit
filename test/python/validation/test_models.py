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

"""Models tests."""

from datetime import datetime

from qiskit.validation import fields
from qiskit.validation.base import BaseModel, BaseSchema, bind_schema
from qiskit.validation.exceptions import ModelValidationError
from qiskit.test import QiskitTestCase


class DummySchema(BaseSchema):
    """Example Dummy schema."""
    pass


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


@bind_schema(DummySchema)
class NotAPerson(BaseModel):
    """Example of NotAPerson model."""
    pass


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
        with self.assertRaises(ModelValidationError):
            _ = Person()

        # From dict.
        with self.assertRaises(ModelValidationError):
            _ = Person.from_dict({})

    def test_instantiate_wrong_type(self):
        """Test model instantiation with fields of the wrong type."""
        with self.assertRaises(ModelValidationError):
            _ = Person(name=1)

        # From dict.
        with self.assertRaises(ModelValidationError):
            _ = Person.from_dict({'name': 1})

    def test_instantiate_deserialized_types(self):
        """Test model instantiation with fields of deserialized type."""
        birth_date = datetime(2000, 1, 1).date()

        person = Person(name='Foo', birth_date=birth_date)
        self.assertEqual(person.birth_date, birth_date)
        with self.assertRaises(ModelValidationError):
            _ = Person(name='Foo', birth_date=birth_date.isoformat())

        # From dict.
        person_dict = Person.from_dict({'name': 'Foo',
                                        'birth_date': birth_date.isoformat()})
        self.assertEqual(person_dict.birth_date, birth_date)
        with self.assertRaises(ModelValidationError):
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
        with self.assertRaises(ModelValidationError):
            _ = Book(title='A Book', author=NotAPerson())

        # From dict.
        with self.assertRaises(ModelValidationError):
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
