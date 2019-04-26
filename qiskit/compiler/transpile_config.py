# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Models for TranspileConfig and its related components."""

from qiskit.compiler.models import TranspileConfigSchema
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
        skip_numeric_passes (bool): if True, do not apply passes containing
            numerical optimization.
    """
    def __init__(self, optimization_level, skip_numeric_passes, **kwargs):
        self.optimization_level = optimization_level
        self.skip_numeric_passes = skip_numeric_passes
        super().__init__(**kwargs)
