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

"""Models for TranspileConfig and its related components."""

from qiskit.transpiler.models import TranspileConfigSchema
from qiskit.validation import BaseModel, bind_schema


@bind_schema(TranspileConfigSchema)
class TranspileConfig(BaseModel):
    """Model for TranspileConfig.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``TranspileConfigSchema``.

    Attributes:
        optimization_level (int): a non-negative integer indicating the
            optimization level. 0 means no transformation on the circuit. Higher
            levels may produce more optimized circuits, but may take longer.
    """
    def __init__(self, optimization_level, **kwargs):
        self.optimization_level = optimization_level
        super().__init__(**kwargs)
