# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Models for Qobj and its related components."""

from marshmallow.validate import Length, OneOf, Range, Regexp

from qiskit.validation.base import BaseModel, BaseSchema, bind_schema
from qiskit.validation.fields import Integer, List, Nested, Raw, String

# Current version of the Qobj schema.
QOBJ_VERSION = '1.1.0'
# Qobj schema versions:
# * 1.1.0: Qiskit 0.8
# * 1.0.0: Qiskit 0.6
# * 0.0.1: Qiskit 0.5.x format (pre-schemas).


class QobjConditionalSchema(BaseSchema):
    """Schema for QobjConditional."""

    # Required properties.
    mask = String(required=True, validate=Regexp('^0x([0-9A-Fa-f])+$'))
    type = String(required=True)
    val = String(required=True, validate=Regexp('^0x([0-9A-Fa-f])+$'))


class QobjInstructionSchema(BaseSchema):
    """Schema for QobjInstruction."""

    # Required properties.
    name = String(required=True)

    # Optional properties.
    qubits = List(Integer(validate=Range(min=0)),
                  validate=Length(min=1))
    params = List(Raw())
    memory = List(Integer(validate=Range(min=0)),
                  validate=Length(min=1))
    conditional = Nested(QobjConditionalSchema)


class QobjExperimentHeaderSchema(BaseSchema):
    """Schema for QobjExperimentHeader."""
    pass


class QobjExperimentConfigSchema(BaseSchema):
    """Schema for QobjConfig."""

    # Required properties.

    # Optional properties.
    memory_slots = Integer(validate=Range(min=0))
    n_qubits = Integer(validate=Range(min=1))


class QobjExperimentSchema(BaseSchema):
    """Schema for QobjExperiment."""

    # Required properties.
    instructions = Nested(QobjInstructionSchema, required=True, many=True,
                          validate=Length(min=1))

    # Optional properties.
    header = Nested(QobjExperimentHeaderSchema)
    config = Nested(QobjExperimentConfigSchema)


class QobjConfigSchema(BaseSchema):
    """Schema for QobjConfig."""

    # Required properties.

    # Optional properties.
    max_credits = Integer()
    seed = Integer()
    memory_slots = Integer(validate=Range(min=0))
    n_qubits = Integer(validate=Range(min=1))
    shots = Integer(validate=Range(min=1))


class QobjHeaderSchema(BaseSchema):
    """Schema for QobjHeader."""

    # Required properties.

    # Optional properties.
    backend_name = String()
    backend_version = String()


class QobjSchema(BaseSchema):
    """Schema for Qobj."""

    # Required properties.
    qobj_id = String(required=True)
    config = Nested(QobjConfigSchema, required=True)
    experiments = Nested(QobjExperimentSchema, required=True, many=True)
    header = Nested(QobjHeaderSchema, required=True)
    type = String(required=True, validate=OneOf(['QASM', 'PULSE']))
    schema_version = String(required=True, missing=QOBJ_VERSION)


@bind_schema(QobjConditionalSchema)
class QobjConditional(BaseModel):
    """Model for QobjConditional.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QobjConditionalSchema``.

    Attributes:
        mask (str): hexadecimal mask of the conditional
        type (str): type of the conditional
        val (str): hexadecimal value of the conditional
    """
    def __init__(self, mask, type, val, **kwargs):
        self.mask = mask
        self.type = type
        self.val = val

        super().__init__(**kwargs)


@bind_schema(QobjInstructionSchema)
class QobjInstruction(BaseModel):
    """Model for QobjInstruction.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QobjInstructionSchema``.

    Attributes:
        name (str): name of the instruction
    """
    def __init__(self, name, **kwargs):
        self.name = name

        super().__init__(**kwargs)


@bind_schema(QobjExperimentHeaderSchema)
class QobjExperimentHeader(BaseModel):
    """Model for QobjExperimentHeader.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QobjExperimentHeaderSchema``.
    """
    pass


@bind_schema(QobjExperimentConfigSchema)
class QobjExperimentConfig(BaseModel):
    """Model for QobjExperimentConfig.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QobjExperimentConfigSchema``.
    """
    pass


@bind_schema(QobjExperimentSchema)
class QobjExperiment(BaseModel):
    """Model for QobjExperiment.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QobjExperimentSchema``.

    Attributes:
        instructions (list[QobjInstruction]): list of instructions.
    """
    def __init__(self, instructions, **kwargs):
        self.instructions = instructions

        super().__init__(**kwargs)


@bind_schema(QobjConfigSchema)
class QobjConfig(BaseModel):
    """Model for QobjConfig.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QobjConfigSchema``.
    """
    pass


@bind_schema(QobjHeaderSchema)
class QobjHeader(BaseModel):
    """Model for QobjHeader.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QobjHeaderSchema``.
    """
    pass


@bind_schema(QobjSchema)
class Qobj(BaseModel):
    """Model for Qobj.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QobjSchema``.

    Attributes:
        qobj_id (str): Qobj identifier.
        config (QobjConfig): config settings for the Qobj.
        experiments (list[QobjExperiment]): list of experiments.
        header (QobjHeader): headers.
        type (str): experiment type (QASM/PULSE).
        schema_version (str): Qobj version.
    """
    def __init__(self, qobj_id, config, experiments, header, type, **kwargs):
        self.qobj_id = qobj_id
        self.config = config
        self.experiments = experiments
        self.header = header
        self.type = type

        self.schema_version = QOBJ_VERSION

        super().__init__(**kwargs)
