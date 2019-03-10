# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Models for RunConfig and its related components."""

from marshmallow.validate import Range

from qiskit.validation import BaseModel, BaseSchema, bind_schema
from qiskit.validation.fields import Boolean, Integer


class RunConfigSchema(BaseSchema):
    """Schema for RunConfig."""

    # Required properties.
    # None

    # Optional properties.
    shots = Integer(validate=Range(min=1))
    max_credits = Integer(validate=Range(min=3, max=10))  # TODO: can we check the range
    seed = Integer()
    memory = Boolean()  # set default to be False


@bind_schema(RunConfigSchema)
class RunConfig(BaseModel):
    """Model for RunConfig.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``RunConfigSchema``.

    Attributes:
        shots (int): the number of shots.
        max_credits (int): the max_credits to use on the IBMQ public devices.
        seed (int): the seed to use in the simulator for the first experiment.
        memory (bool): to use memory.
    """
