# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""The qasm qobj models."""

from marshmallow.validate import Range, Length, Regexp

from qiskit.validation import bind_schema, BaseSchema, BaseModel
from qiskit.validation.fields import List, Integer, InstructionParameter, Nested, String
from .base import (QobjInstructionSchema, QobjExperimentConfigSchema, QobjExperimentSchema,
                   QobjConfigSchema, QobjInstruction, QobjExperimentConfig,
                   QobjExperiment, QobjConfig)


class QobjConditionalSchema(BaseSchema):
    """Schema for QobjConditional."""

    # Required properties.
    mask = String(required=True, validate=Regexp('^0x([0-9A-Fa-f])+$'))
    type = String(required=True)
    val = String(required=True, validate=Regexp('^0x([0-9A-Fa-f])+$'))


class QasmQobjInstructionSchema(QobjInstructionSchema):
    """Schema for QasmQobjInstruction."""

    # Optional properties.
    qubits = List(Integer(validate=Range(min=0)),
                  validate=Length(min=1))
    params = List(InstructionParameter())
    memory = List(Integer(validate=Range(min=0)),
                  validate=Length(min=1))
    conditional = Nested(QobjConditionalSchema)


class QasmQobjExperimentConfigSchema(QobjExperimentConfigSchema):
    """Schema for QasmQobjExperimentConfig."""

    # Optional properties.
    memory_slots = Integer(validate=Range(min=0))
    n_qubits = Integer(validate=Range(min=1))


class QasmQobjExperimentSchema(QobjExperimentSchema):
    """Schema for QasmQobjExperiment."""

    # Required properties.
    instructions = Nested(QasmQobjInstructionSchema, required=True, many=True,
                          validate=Length(min=1))

    # Optional properties.
    config = Nested(QasmQobjExperimentConfigSchema)


class QasmQobjConfigSchema(QobjConfigSchema):
    """Schema for QasmQobjConfig."""

    # Optional properties.
    n_qubits = Integer(validate=Range(min=1))


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
        # pylint: disable=redefined-builtin
        self.mask = mask
        self.type = type
        self.val = val

        super().__init__(**kwargs)


@bind_schema(QasmQobjInstructionSchema)
class QasmQobjInstruction(QobjInstruction):
    """Model for QasmQobjInstruction inherit from QobjInstruction.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QasmQobjInstructionSchema``.

    Attributes:
        name (str): name of the instruction
    """
    def __init__(self, name, **kwargs):
        super().__init__(name=name,
                         **kwargs)


@bind_schema(QasmQobjExperimentConfigSchema)
class QasmQobjExperimentConfig(QobjExperimentConfig):
    """Model for QasmQobjExperimentConfig inherit from QobjExperimentConfig.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QasmQobjExperimentConfigSchema``.
    """
    pass


@bind_schema(QasmQobjExperimentSchema)
class QasmQobjExperiment(QobjExperiment):
    """Model for QasmQobjExperiment inherit from QobjExperiment.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QasmQobjExperimentSchema``.

    Attributes:
        instructions (list[QasmQobjInstruction]): list of instructions.
    """
    def __init__(self, instructions, **kwargs):
        super().__init__(instructions=instructions,
                         **kwargs)


@bind_schema(QasmQobjConfigSchema)
class QasmQobjConfig(QobjConfig):
    """Model for QasmQobjConfig inherit from QobjConfig.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QasmQobjConfigSchema``.
    """
    pass
