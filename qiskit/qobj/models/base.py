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

"""The generic qobj models."""

from marshmallow.validate import Range

from qiskit.validation import BaseSchema, bind_schema, BaseModel
from qiskit.validation.fields import String, Nested, Integer


class QobjInstructionSchema(BaseSchema):
    """Base Schema for QobjInstruction."""

    # Required properties
    name = String(required=True)


class QobjExperimentHeaderSchema(BaseSchema):
    """Base Schema for QobjExperimentHeader."""
    pass


class QobjExperimentConfigSchema(BaseSchema):
    """Base Schema for QobjExperimentConfig."""
    pass


class QobjExperimentSchema(BaseSchema):
    """Base Schema for QobjExperiment."""

    # Required properties.
    instructions = Nested(QobjInstructionSchema, required=True, many=True)

    # Optional properties.
    header = Nested(QobjExperimentHeaderSchema)
    config = Nested(QobjExperimentConfigSchema)


class QobjConfigSchema(BaseSchema):
    """Base Schema for QobjConfig."""

    # Optional properties.
    max_credits = Integer()
    seed_simulator = Integer()
    memory_slots = Integer(validate=Range(min=0))
    shots = Integer(validate=Range(min=1))


class QobjHeaderSchema(BaseSchema):
    """Base Schema for QobjHeader."""

    # Optional properties.
    backend_name = String()
    backend_version = String()


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
