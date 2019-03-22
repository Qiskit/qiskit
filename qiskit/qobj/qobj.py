# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Model for Qobj."""

from marshmallow.validate import Equal, OneOf

from qiskit.validation.base import BaseModel, BaseSchema, bind_schema
from qiskit.validation.fields import Nested, String

from .models import (BaseQobjConfigSchema, BaseQobjExperimentSchema, BaseQobjHeaderSchema,
                     QASMQobjConfigSchema, QASMQobjExperimentSchema, QASMQobjHeaderSchema,
                     PulseQobjConfigSchema, PulseQobjExperimentSchema, PulseQobjHeaderSchema)
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
    schema_version = String(required=True, missing=QOBJ_VERSION)

    # Required properties depend on Qobj type.
    config = Nested(BaseQobjConfigSchema, required=True)
    experiments = Nested(BaseQobjExperimentSchema, required=True, many=True)
    header = Nested(BaseQobjHeaderSchema, required=True)
    type = String(required=True, validate=OneOf(QobjType.QASM, QobjType.PULSE))


class QASMQobjSchema(QobjSchema):
    """Schema for QASMQobj."""

    # Required properties.
    config = Nested(QASMQobjConfigSchema, required=True)
    experiments = Nested(QASMQobjExperimentSchema, required=True, many=True)
    header = Nested(QASMQobjHeaderSchema, required=True)

    type = String(validate=Equal(QobjType.QASM))


class PulseQobjSchema(QobjSchema):
    """Schema for PulseQobj."""

    # Required properties.
    config = Nested(PulseQobjConfigSchema, required=True)
    experiments = Nested(PulseQobjExperimentSchema, required=True, many=True)
    header = Nested(PulseQobjHeaderSchema, required=True)

    type = String(validate=Equal(QobjType.PULSE))


@bind_schema(QobjSchema)
class Qobj(BaseModel):
    """Model for Qobj.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QobjSchema``.

    Attributes:
        qobj_id (str): Qobj identifier.
        config (QASMQobjConfig): config settings for the Qobj.
        experiments (list[QASMQobjExperiment]): list of experiments.
        header (QASMQobjHeader): headers.
        type (str): Qobj type.
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


@bind_schema(QASMQobjSchema)
class QASMQobj(Qobj):
    """Model for QASMQobj inherit from Qobj.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QASMQobjSchema``.

    Attributes:
        qobj_id (str): Qobj identifier.
        config (QASMQobjConfig): config settings for the Qobj.
        experiments (list[QASMQobjExperiment]): list of experiments.
        header (QASMQobjHeader): headers.
    """
    def __init__(self, qobj_id, config, experiments, header, **kwargs):

        # to avoid specifying 'type' here within from_dict()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'type'}

        super().__init__(qobj_id=qobj_id,
                         config=config,
                         experiments=experiments,
                         header=header,
                         type='QASM',
                         **filtered_kwargs)


@bind_schema(PulseQobjSchema)
class PulseQobj(Qobj):
    """Model for PulseQobj inherit from Qobj.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``PulseQobjSchema``.

    Attributes:
        qobj_id (str): Qobj identifier.
        config (PulseQobjConfig): config settings for the Qobj.
        experiments (list[PulseQobjExperiment]): list of experiments.
        header (PulseQobjHeader): headers.
    """
    def __init__(self, qobj_id, config, experiments, header, **kwargs):

        # to avoid specifying 'type' here within from_dict()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'type'}

        super().__init__(qobj_id=qobj_id,
                         config=config,
                         experiments=experiments,
                         header=header,
                         type='PULSE',
                         **filtered_kwargs)
