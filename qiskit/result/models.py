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

"""Schema and helper models for schema-conformant Results."""

from marshmallow.validate import Length, OneOf, Regexp, Range

from qiskit.validation.base import BaseModel, BaseSchema, ObjSchema, bind_schema
from qiskit.validation.fields import Complex, ByType
from qiskit.validation.fields import Boolean, DateTime, Integer, List, Nested
from qiskit.validation.fields import Raw, String, NumpyArray
from qiskit.validation.validate import PatternProperties
from qiskit.qobj.utils import MeasReturnType, MeasLevel


class ExperimentResultDataSchema(BaseSchema):
    """Schema for ExperimentResultData."""

    counts = Nested(ObjSchema,
                    validate=PatternProperties(
                        {Regexp('^0x([0-9A-Fa-f])+$'): Integer()}))
    snapshots = Nested(ObjSchema)
    memory = List(Raw(),
                  validate=Length(min=1))
    statevector = NumpyArray(Complex(),
                             validate=Length(min=1))
    unitary = NumpyArray(NumpyArray(Complex(),
                                    validate=Length(min=1)),
                         validate=Length(min=1))


class ExperimentResultSchema(BaseSchema):
    """Schema for ExperimentResult."""

    # Required fields.
    shots = ByType([Integer(), List(Integer(validate=Range(min=1)),
                                    validate=Length(equal=2))],
                   required=True)
    success = Boolean(required=True)
    data = Nested(ExperimentResultDataSchema, required=True)

    # Optional fields.
    status = String()
    seed = Integer()
    meas_level = Integer(validate=OneOf(choices=(MeasLevel.RAW,
                                                 MeasLevel.KERNELED,
                                                 MeasLevel.CLASSIFIED)))
    meas_return = String(validate=OneOf(choices=(MeasReturnType.AVERAGE,
                                                 MeasReturnType.SINGLE)))
    header = Nested(ObjSchema)


class ResultSchema(BaseSchema):
    """Schema for Result."""

    # Required fields.
    backend_name = String(required=True)
    backend_version = String(required=True,
                             validate=Regexp('[0-9]+.[0-9]+.[0-9]+$'))
    qobj_id = String(required=True)
    job_id = String(required=True)
    success = Boolean(required=True)
    results = Nested(ExperimentResultSchema, required=True, many=True)

    # Optional fields.
    date = DateTime()
    status = String()
    header = Nested(ObjSchema)


@bind_schema(ExperimentResultDataSchema)
class ExperimentResultData(BaseModel):
    """Model for ExperimentResultData.

    Please note that this class only describes the required fields. For the
    full description of the model, please check
    ``ExperimentResultDataSchema``.
    """
    pass


@bind_schema(ExperimentResultSchema)
class ExperimentResult(BaseModel):
    """Model for ExperimentResult.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``ExperimentResultSchema``.

    Attributes:
        shots (int or tuple): the starting and ending shot for this data.
        success (bool): if true, we can trust results for this experiment.
        data (ExperimentResultData): results information.
        meas_level (int): Measurement result level.
    """

    def __init__(self, shots, success, data, meas_level=MeasLevel.CLASSIFIED, **kwargs):
        self.shots = shots
        self.success = success
        self.data = data
        self.meas_level = meas_level

        super().__init__(**kwargs)
