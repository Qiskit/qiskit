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

"""The qasm qobj models."""

from marshmallow.validate import Range, Length

from qiskit.validation import bind_schema
from qiskit.validation.fields import List, Integer, InstructionParameter, Nested
from .base import (QobjInstructionSchema, QobjExperimentConfigSchema, QobjExperimentSchema,
                   QobjConfigSchema, QobjInstruction, QobjExperimentConfig,
                   QobjExperiment, QobjConfig)


class QasmQobjInstructionSchema(QobjInstructionSchema):
    """Schema for QasmQobjInstruction."""

    # Optional properties.
    qubits = List(Integer(validate=Range(min=0)),
                  validate=Length(min=1))
    params = List(InstructionParameter())
    memory = List(Integer(validate=Range(min=0)),
                  validate=Length(min=1))
    conditional = Integer(validate=Range(min=0))


class QasmQobjExperimentConfigSchema(QobjExperimentConfigSchema):
    """Schema for QasmQobjExperimentConfig."""

    # Optional properties.
    memory_slots = Integer(validate=Range(min=0))
    n_qubits = Integer(validate=Range(min=1))


class QasmQobjExperimentSchema(QobjExperimentSchema):
    """Schema for QasmQobjExperiment."""

    # Required properties.
    instructions = Nested(QasmQobjInstructionSchema, required=True, many=True)

    # Optional properties.
    config = Nested(QasmQobjExperimentConfigSchema)


class QasmQobjConfigSchema(QobjConfigSchema):
    """Schema for QasmQobjConfig."""

    # Optional properties.
    n_qubits = Integer(validate=Range(min=1))


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
