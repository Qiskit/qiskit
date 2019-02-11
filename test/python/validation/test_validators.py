# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test qiskit validators."""

from marshmallow.validate import Regexp

from qiskit.validation import fields, ModelValidationError
from qiskit.validation.base import BaseModel, BaseSchema, bind_schema, ObjSchema, Obj
from qiskit.validation.validate import PatternProperties
from qiskit.test import QiskitTestCase


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
        with self.assertRaises(ModelValidationError):
            _ = Histogram(**invalid_key_data)

        # From dict
        with self.assertRaises(ModelValidationError):
            _ = Histogram.from_dict(invalid_key_data)

    def test_patternproperties_invalid_value(self):
        """Test the PatternProperties validator fails when invalid value"""
        invalid_value_data = {'counts': {'0x00': 'so many', '0x11': 50}}
        with self.assertRaises(ModelValidationError):
            _ = Histogram(**invalid_value_data)

        # From dict
        with self.assertRaises(ModelValidationError):
            _ = Histogram.from_dict(invalid_value_data)

    def test_patternproperties_to_dict(self):
        """Test a field using the PatternProperties validator produces a correct value"""
        counts_dict = {'0x00': 50, '0x11': 50}
        counts = Obj(**counts_dict)
        histogram = Histogram(counts=counts)
        histogram_dict = histogram.to_dict()
        self.assertEqual(histogram_dict, {'counts': counts_dict})
