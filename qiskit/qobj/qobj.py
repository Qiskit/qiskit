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

"""Model for Qobj."""

from marshmallow.validate import Equal, OneOf

from qiskit.qobj.models.base import QobjExperimentSchema, QobjConfigSchema, QobjHeaderSchema
from qiskit.qobj.models.pulse import PulseQobjExperimentSchema, PulseQobjConfigSchema
from qiskit.qobj.models.qasm import QasmQobjExperimentSchema, QasmQobjConfigSchema
from qiskit.validation.base import BaseModel, BaseSchema, bind_schema
from qiskit.validation.fields import Nested, String
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
    config = Nested(QobjConfigSchema, required=True)
    experiments = Nested(QobjExperimentSchema, required=True, many=True)
    header = Nested(QobjHeaderSchema, required=True)
    type = String(required=True, validate=OneOf(choices=(QobjType.QASM, QobjType.PULSE)))


class QasmQobjSchema(QobjSchema):
    """Schema for QasmQobj."""

    # Required properties.
    config = Nested(QasmQobjConfigSchema, required=True)
    experiments = Nested(QasmQobjExperimentSchema, required=True, many=True)

    type = String(required=True, validate=Equal(QobjType.QASM),
                  missing=QobjType.QASM)


class PulseQobjSchema(QobjSchema):
    """Schema for PulseQobj."""

    # Required properties.
    config = Nested(PulseQobjConfigSchema, required=True)
    experiments = Nested(PulseQobjExperimentSchema, required=True, many=True)

    type = String(required=True, validate=Equal(QobjType.PULSE),
                  missing=QobjType.PULSE)


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


@bind_schema(QasmQobjSchema)
class QasmQobj(Qobj):
    """Model for QasmQobj inherit from Qobj.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QasmQobjSchema``.

    Attributes:
        qobj_id (str): Qobj identifier.
        config (QASMQobjConfig): config settings for the Qobj.
        experiments (list[QASMQobjExperiment]): list of experiments.
        header (QobjHeader): headers.
    """
    def __init__(self, qobj_id, config, experiments, header, **kwargs):

        # to avoid specifying 'type' here within from_dict()
        kwargs.pop('type', None)

        super().__init__(qobj_id=qobj_id,
                         config=config,
                         experiments=experiments,
                         header=header,
                         type=QobjType.QASM.value,
                         **kwargs)


@bind_schema(PulseQobjSchema)
class PulseQobj(Qobj):
    """Model for PulseQobj inherit from Qobj.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``PulseQobjSchema``.

    Attributes:
        qobj_id (str): Qobj identifier.
        config (PulseQobjConfig): config settings for the Qobj.
        experiments (list[PulseQobjExperiment]): list of experiments.
        header (QobjHeader): headers.
    """
    def __init__(self, qobj_id, config, experiments, header, **kwargs):

        # to avoid specifying 'type' here within from_dict()
        kwargs.pop('type', None)

        super().__init__(qobj_id=qobj_id,
                         config=config,
                         experiments=experiments,
                         header=header,
                         type=QobjType.PULSE.value,
                         **kwargs)
