# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Models for RunConfig and its related components."""

from marshmallow.validate import Range

from qiskit.validation import BaseSchema
from qiskit.validation.fields import Boolean, Integer


class TranspileConfigSchema(BaseSchema):
    """Schema for TranspileConfig."""

    # Required properties.
    # None

    # Optional properties.


class RunConfigSchema(BaseSchema):
    """Schema for RunConfig."""

    # Required properties.
    # None

    # Optional properties.
    shots = Integer(validate=Range(min=1))
    max_credits = Integer(validate=Range(min=3, max=10))  # TODO: can we check the range
    seed = Integer()
    memory = Boolean()  # set default to be False
