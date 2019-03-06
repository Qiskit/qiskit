# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Models for Qobj and its related components."""


from marshmallow.validate import Range, Regexp, OneOf, Length

from qiskit.validation import BaseModel, BaseSchema, bind_schema
from qiskit.validation.fields import Boolean, Integer, String, Nested


class QobjInstruction(BaseSchema):
    """Schema for QobjInstruction."""

    # Required properties.
    name = String(required=True)

    # Optional properties.
    # TODO: solve ambiguities


class QobjExperimentHeader(BaseSchema):
    """Schema for QobjExperimentHeader."""
    pass


class QobjExperimentConfig(BaseSchema):
    """Schema for QobjConfig."""

    # Required properties.

    # Optional properties.
    memory_slots = Integer(validate=Range(min=0))
    n_qubits = Integer(validate=Range(min=1))


class QobjExperiment(BaseSchema):
    """Schema for QobjExperiment."""

    # Required properties.
    instructions = Nested(QobjInstruction, required=True, many=True,
                          validate=Length(min=1))

    # Optional properties.
    header = Nested(QobjExperimentHeader)
    config = Nested(QobjExperimentConfig)


class QobjConfig(BaseSchema):
    """Schema for QobjConfig."""

    # Required properties.

    # Optional properties.
    max_credits = Integer()
    seed = Integer()
    memory_slots = Integer(validate=Range(min=0))
    n_qubits = Integer(validate=Range(min=1))
    shots = Integer(validate=Range(min=1))


class QobjHeader(BaseSchema):
    """Schema for QobjHeader."""

    # Required properties.

    # Optional properties.
    backend_name = String()
    backend_version = String()


class QobjSchema(BaseSchema):
    """Schema for Qobj."""

    # Required properties.
    qobj_id = String(required=True)
    config = Nested(QobjConfig, required=True)
    experiments = None
    header = Nested(QobjHeader, required=True)
    type = String(required=True, validate=OneOf(['QASM', 'PULSE']))
    schema_version = String(required=True)


@bind_schema(QobjSchema)
class Qobj(BaseModel):
    """Model for Qobj.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QobjSchema``.

    Attributes:
        backend_name (str): backend name.
        backend_version (str): backend version in the form X.Y.Z.
        operational (bool): backend operational and accepting jobs.
        pending_jobs (int): number of pending jobs on the backend.
        status_msg (str): status message.
    """

    def __init__(self, backend_name, backend_version, operational,
                 pending_jobs, status_msg, **kwargs):
        self.backend_name = backend_name
        self.backend_version = backend_version
        self.operational = operational
        self.pending_jobs = pending_jobs
        self.status_msg = status_msg

        super().__init__(**kwargs)
