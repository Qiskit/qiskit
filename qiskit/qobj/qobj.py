# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Model for Qobj."""

from marshmallow.validate import OneOf

from qiskit.validation.base import BaseModel, BaseSchema, bind_schema
from qiskit.validation.fields import Nested, String

from .models import QobjConfigSchema, QobjExperimentSchema, QobjHeaderSchema
from .utils import QobjType


QOBJ_VERSION = '1.1.0'
"""Current version of the Qobj schema.

Qobj schema versions:
* 1.1.0: Qiskit Terra 0.8
* 1.0.0: Qiskit Terra 0.6
* 0.0.1: Qiskit Terra 0.5.x format (pre-schemas).
"""


class QobjSchema(BaseSchema):
    """Schema for Qobj."""

    # Required properties.
    qobj_id = String(required=True)
    config = Nested(QobjConfigSchema, required=True)
    experiments = Nested(QobjExperimentSchema, required=True, many=True)
    header = Nested(QobjHeaderSchema, required=True)
    type = String(required=True,
                  validate=OneOf(QobjType.QASM.value,
                                 QobjType.PULSE.value))
    schema_version = String(required=True, missing=QOBJ_VERSION)


@bind_schema(QobjSchema)
class Qobj(BaseModel):
    """Model for Qobj.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QobjSchema``.

    Attributes:
        qobj_id (str): Qobj identifier.
        config (QobjConfig): config settings for the Qobj.
        experiments (list[QobjExperiment]): list of experiments.
        header (QobjHeader): headers.
        type (str): experiment type (QASM/PULSE).
        schema_version (str): Qobj version.
    """
    def __init__(self, qobj_id, config, experiments, header, type, **kwargs):
        # pylint: disable=redefined-builtin
        self.qobj_id = qobj_id
        self.config = config
        self.experiments = experiments
        self.header = header
        self.type = type

        self.schema_version = QOBJ_VERSION

        super().__init__(**kwargs)
