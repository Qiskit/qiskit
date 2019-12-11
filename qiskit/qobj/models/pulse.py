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

"""The pulse qobj models."""

from marshmallow.validate import Range, Regexp, Length

from qiskit.qobj.utils import MeasReturnType, MeasLevel
from qiskit.validation import BaseSchema, bind_schema, BaseModel
from qiskit.validation.fields import (Integer, String, Number, Float, Complex, List,
                                      Nested, DictParameters, ByType)
from .base import QobjInstructionSchema


class QobjMeasurementOptionSchema(BaseSchema):
    """Schema for QobjMeasOptiton."""

    # Required properties.
    name = String(required=True)
    params = DictParameters(valid_value_types=(int, float, str, bool, type(None)),
                            required=True)


class PulseLibraryItemSchema(BaseSchema):
    """Schema for PulseLibraryItem."""

    # Required properties.
    name = String(required=True)
    samples = List(Complex(), required=True, validate=Length(min=1))


class PulseQobjInstructionSchema(QobjInstructionSchema):
    """Schema for PulseQobjInstruction."""
    # pylint: disable=invalid-name

    # Required properties
    t0 = Integer(required=True, validate=Range(min=0))

    # Optional properties.
    ch = String(validate=Regexp('[dum]([0-9])+'))
    conditional = Integer(validate=Range(min=0))
    val = ByType([Complex(), String()])
    phase = ByType([Float(), String()])
    duration = Integer(validate=Range(min=1))
    qubits = List(Integer(validate=Range(min=0)), validate=Length(min=1))
    memory_slot = List(Integer(validate=Range(min=0)), validate=Length(min=1))
    register_slot = List(Integer(validate=Range(min=0)), validate=Length(min=1))
    kernels = Nested(QobjMeasurementOptionSchema, many=True)
    discriminators = Nested(QobjMeasurementOptionSchema, many=True)
    label = String()
    type = String()
