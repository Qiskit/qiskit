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

"""Models for RunConfig and its related components."""

from qiskit.assembler.models import RunConfigSchema
from qiskit.validation import BaseModel, bind_schema


@bind_schema(RunConfigSchema)
class RunConfig(BaseModel):
    """Model for RunConfig.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``RunConfigSchema``.

    Attributes:
        shots (int): the number of shots
        max_credits (int): the max_credits to use on the IBM Q public devices
        seed_simulator (int): the seed to use in the simulator
        memory (bool): whether to request memory from backend (per-shot readouts)
        parameter_binds (list[dict]): List of parameter bindings
    """
