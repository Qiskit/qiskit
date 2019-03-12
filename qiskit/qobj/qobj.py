# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Model for Qobj."""

from marshmallow.validate import Equal

from qiskit.validation.base import BaseModel, BaseSchema, bind_schema
from qiskit.validation.fields import Nested, String

from .models import QASMQobjConfigSchema, QASMQobjExperimentSchema, QASMQobjHeaderSchema
from .models import PulseQobjConfigSchema, PulseQobjExperimentSchema, PulseQobjHeaderSchema
from .utils import QobjType


QOBJ_VERSION = '1.1.0'
"""Current version of the Qobj schema.

Qobj schema versions:
* 1.1.0: Qiskit Terra 0.8
* 1.0.0: Qiskit Terra 0.6
* 0.0.1: Qiskit Terra 0.5.x format (pre-schemas).
"""


class QASMQobjSchema(BaseSchema):
    """Schema for QASMQobj."""

    # Required properties.
    qobj_id = String(required=True)
    config = Nested(QASMQobjConfigSchema, required=True)
    experiments = Nested(QASMQobjExperimentSchema, required=True, many=True)
    header = Nested(QASMQobjHeaderSchema, required=True)
    type = String(validate=Equal(QobjType.QASM))
    schema_version = String(required=True, missing=QOBJ_VERSION)


class PulseQobjSchema(BaseSchema):
    """Schema for PulseQobj."""

    # Required properties.
    qobj_id = String(required=True)
    config = Nested(PulseQobjConfigSchema, required=True)
    experiments = Nested(PulseQobjExperimentSchema, required=True, many=True)
    header = Nested(PulseQobjHeaderSchema, required=True)
    type = String(validate=Equal(QobjType.PULSE))
    schema_version = String(required=True, missing=QOBJ_VERSION)


@bind_schema(QASMQobjSchema)
class QASMQobj(BaseModel):
    """Model for QASMQobj.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QobjSchema``.

    Attributes:
        qobj_id (str): Qobj identifier.
        config (QASMQobjConfig): config settings for the Qobj.
        experiments (list[QASMQobjExperiment]): list of experiments.
        header (QASMQobjHeader): headers.
        schema_version (str): Qobj version.
    """
    def __init__(self, qobj_id, config, experiments, header, **kwargs):
        # pylint: disable=redefined-builtin
        self.qobj_id = qobj_id
        self.config = config
        self.experiments = experiments
        self.header = header
        self.type = 'QASM'

        self.schema_version = QOBJ_VERSION

        super().__init__(**kwargs)


@bind_schema(PulseQobjSchema)
class PulseQobj(BaseModel):
    """Model for PulseQobj.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``PulseQobjSchema``.

    Attributes:
        qobj_id (str): Qobj identifier.
        config (PulseQobjConfig): config settings for the Qobj.
        experiments (list[PulseQobjExperiment]): list of experiments.
        header (PulseQobjHeader): headers.
        schema_version (str): Qobj version.
    """
    def __init__(self, qobj_id, config, experiments, header, **kwargs):
        # pylint: disable=redefined-builtin
        self.qobj_id = qobj_id
        self.config = config
        self.experiments = experiments
        self.header = header
        self.type = 'PULSE'

        self.schema_version = QOBJ_VERSION

        super().__init__(**kwargs)
