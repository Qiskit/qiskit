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

"""Test polymorphic and validated fields."""

from qiskit.validation import fields
from qiskit.validation.base import BaseModel, BaseSchema, bind_schema
from qiskit.validation.exceptions import ModelValidationError
from qiskit.validation.fields import TryFrom, ByAttribute, ByType
from qiskit.test import QiskitTestCase


class DummySchema(BaseSchema):
    """Simple Dummy schema."""
    pass


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


@bind_schema(DummySchema)
class NotAPet(BaseModel):
    """Simple NotAPet model."""
    pass


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
        with self.assertRaises(ModelValidationError) as context_manager:
            _ = PetOwner(auto_pets=[Cat(fur_density=1.5), NotAPet()])
        self.assertIn('auto_pets', str(context_manager.exception))

    def test_try_from_field_invalid_from_dict(self):
        """Test the TryFrom field, instantiation from dict, with invalid."""
        with self.assertRaises(ModelValidationError) as context_manager:
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
        with self.assertRaises(ModelValidationError) as context_manager:
            _ = PetOwner(by_attribute_pets=[Cat(fur_density=1.5), NotAPet()])
        self.assertIn('by_attribute_pets', str(context_manager.exception))

    def test_by_attribute_field_invalid_from_dict(self):
        """Test the ByAttribute field, instantiation from dict, with invalid."""
        with self.assertRaises(ModelValidationError) as context_manager:
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
        with self.assertRaises(ModelValidationError) as context_manager:
            _ = PetOwner(by_type_contact=123)
        self.assertIn('by_type_contact', str(context_manager.exception))

    def test_by_type_field_invalid_from_dict(self):
        """Test the ByType field, instantiation from dict, with invalid."""
        with self.assertRaises(ModelValidationError) as context_manager:
            _ = PetOwner.from_dict({'by_type_contact': 123})
        self.assertIn('by_type_contact', str(context_manager.exception))
