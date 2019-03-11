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

    """
