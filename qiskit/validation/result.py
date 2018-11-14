# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Schema and Model for schema-conformant Results."""

from marshmallow.fields import Boolean, DateTime, Integer, List, Nested, Raw, String
from marshmallow.validate import Length, OneOf, Regexp, Range

from qiskit.validation.base import BaseModel, BaseSchema, ObjSchema, bind_schema
from qiskit.validation.fields import Complex, ByType
from qiskit.validation.validate import PatternProperties


class ExperimentResultDataSchema(BaseSchema):
    counts = Nested(ObjSchema,
                    validate=PatternProperties(
                        {Regexp('^0x([0-9A-Fa-f])+$'): Integer()}))
    snapshots = Nested(ObjSchema)
    memory = List(Raw(),
                  validate=Length(min=1))
    statevector = List(List(Complex(),
                            validate=Length(min=1)))
    unitary = List(List(List(Complex(),
                             validate=Length(min=1)),
                        validate=Length(min=1)),
                   validate=Length(min=1))


class ExperimentResultSchema(BaseSchema):
    # Required fields.
    shots = ByType([Integer(), List(Integer(validate=Range(min=1)),
                                    validate=Length(equal=2))],
                   required=True)
    success = Boolean(required=True)
    data = Nested(ExperimentResultDataSchema, required=True)

    # Optional fields.
    status = String()
    seed = Integer()
    meas_return = String(validate=OneOf(['single', 'avg']))
    header = Nested(ObjSchema)


class ResultSchema(BaseSchema):
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
    pass


@bind_schema(ExperimentResultSchema)
class ExperimentResult(BaseModel):
    pass


@bind_schema(ResultSchema)
class Result(BaseModel):
    pass
