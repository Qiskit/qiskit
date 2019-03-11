# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Models for RunConfig and its related components."""

from qiskit.compiler.models import RunConfigSchema
from qiskit.validation import BaseModel, bind_schema


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
