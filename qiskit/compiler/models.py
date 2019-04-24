# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Models for TranspileConfig and RunConfig."""

from marshmallow.validate import Length, Range, OneOf

from qiskit.validation import BaseSchema
from qiskit.validation.fields import Boolean, Integer, List, String, Number, Nested
from qiskit.providers.models import BackendProperties
from qiskit.qobj.models.base import QobjHeader
from qiskit.mapper import Layout, CouplingMap


class TranspileConfigSchema(BaseSchema):
    """Schema for TranspileConfig."""

    # Required properties.
    """
    optimization_level = Integer(validate=Range(min=0, max=2), required=True)
    skip_numeric_passes = Boolean()
    """

    # Optional properties.
    """
    basis_gates = List(String())
    coupling_map = Nested(CouplingMap)
    backend_properties = Nested(BackendProperties, many=True)
    initial_layout = Nested(Layout)
    seed_transpiler = Integer()
    """


class RunConfigSchema(BaseSchema):
    """Schema for RunConfig."""

    # Required properties.
    # None

    # Optional properties.
    """
    qobj_id = String()
    qobj_header = Nested(QobjHeader)
    shots = Integer(validate=Range(min=1))
    memory = Boolean()  # TODO: if True, meas_level MUST be 2
    max_credits = Integer(validate=Range(min=3, max=10))
    seed_simulator = Integer()
    default_qubit_los = List(Number())
    default_meas_los = List(Number())
    schedule_los = List(Number())  # FIXME: double check this with assembler docstring
    meas_level = Integer(required=True, validate=Range(min=0, max=2))
    meas_return = String(validate=OneOf(['single', 'avg']))
    memory_slots = Integer(validate=Range(min=0))
    memory_slot_size = Integer(validate=Range(min=1))
    rep_time = Integer()
    """
