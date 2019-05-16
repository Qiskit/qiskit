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

"""The pulse qobj models."""

from marshmallow.validate import Range, Regexp, Length, OneOf

from qiskit.qobj.utils import MeasReturnType
from qiskit.validation import BaseSchema, bind_schema, BaseModel
from qiskit.validation.fields import (Integer, String, Number, Complex, List,
                                      Nested, DictParameters, ByType)
from .base import (QobjInstructionSchema, QobjExperimentConfigSchema, QobjExperimentSchema,
                   QobjConfigSchema, QobjInstruction, QobjExperimentConfig,
                   QobjExperiment, QobjConfig)


class QobjMeasurementOptionSchema(BaseSchema):
    """Schema for QobjMeasOptiton."""

    # Required properties.
    name = String(required=True)
    params = DictParameters(valid_value_types=(int, float, str, bool, type(None)),
                            required=True)


class PulseLibraryItemSchema(BaseSchema):
    """Schema for PulseLibraryItem."""

    # Required properties.
    name = String(required=True)
    samples = List(Complex(), required=True, validate=Length(min=1))


class PulseQobjInstructionSchema(QobjInstructionSchema):
    """Schema for PulseQobjInstruction."""
    # pylint: disable=invalid-name

    # Required properties
    t0 = Integer(required=True, validate=Range(min=0))

    # Optional properties.
    ch = String(validate=Regexp('[dum]([0-9])+'))
    conditional = Integer(validate=Range(min=0))
    val = ByType([Complex(), String()])
    phase = ByType([Number(), String()])
    duration = Integer(validate=Range(min=1))
    qubits = List(Integer(validate=Range(min=0)), validate=Length(min=1))
    memory_slot = List(Integer(validate=Range(min=0)), validate=Length(min=1))
    register_slot = List(Integer(validate=Range(min=0)), validate=Length(min=1))
    kernels = Nested(QobjMeasurementOptionSchema, many=True)
    discriminators = Nested(QobjMeasurementOptionSchema, many=True)
    label = String()
    type = String()


class PulseQobjExperimentConfigSchema(QobjExperimentConfigSchema):
    """Schema for PulseQobjExperimentConfig."""

    # Optional properties.
    qubit_lo_freq = List(Number())
    meas_lo_freq = List(Number())


class PulseQobjExperimentSchema(QobjExperimentSchema):
    """Schema for PulseQobjExperiment."""

    # Required properties.
    instructions = Nested(PulseQobjInstructionSchema, required=True, many=True,
                          validate=Length(min=1))

    # Optional properties.
    config = Nested(PulseQobjExperimentConfigSchema)


class PulseQobjConfigSchema(QobjConfigSchema):
    """Schema for PulseQobjConfig of device backend."""

    # Required properties.
    meas_level = Integer(required=True, validate=Range(min=0, max=2))
    meas_return = String(required=True, validate=OneOf(choices=(MeasReturnType.AVERAGE,
                                                                MeasReturnType.SINGLE)))
    pulse_library = Nested(PulseLibraryItemSchema, required=True, many=True)
    qubit_lo_freq = List(Number(validate=Range(min=0)), required=True)
    meas_lo_freq = List(Number(validate=Range(min=0)), required=True)

    # Optional properties.
    memory_slot_size = Integer(validate=Range(min=1))
    rep_time = Integer(validate=Range(min=0))


@bind_schema(QobjMeasurementOptionSchema)
class QobjMeasurementOption(BaseModel):
    """Model for QobjMeasurementOption.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``QobjMeasurementOptionSchema``.

    Attributes:
        name (str): name of option specified in the backend
        params (dict): measurement parameter
    """
    def __init__(self, name, params, **kwargs):
        self.name = name
        self.params = params

        super().__init__(**kwargs)


@bind_schema(PulseLibraryItemSchema)
class PulseLibraryItem(BaseModel):
    """Model for PulseLibraryItem.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``PulseLibraryItemSchema``.

    Attributes:
        name (str): name of pulse
        samples (list[complex]]): list of complex values defining pulse shape
    """

    def __init__(self, name, samples, **kwargs):
        self.name = name
        self.samples = samples

        super().__init__(**kwargs)


@bind_schema(PulseQobjInstructionSchema)
class PulseQobjInstruction(QobjInstruction):
    """Model for PulseQobjInstruction inherit from QobjInstruction.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``PulseQobjInstructionSchema``.

    Attributes:
        name (str): name of the instruction
        t0 (int): timing of executing the instruction
    """
    def __init__(self, name, t0, **kwargs):
        # pylint: disable=invalid-name
        self.t0 = t0

        super().__init__(name=name,
                         t0=t0,
                         **kwargs)


@bind_schema(PulseQobjExperimentConfigSchema)
class PulseQobjExperimentConfig(QobjExperimentConfig):
    """Model for PulseQobjExperimentConfig inherit from QobjExperimentConfig.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``PulseQobjExperimentConfigSchema``.
    """
    pass


@bind_schema(PulseQobjExperimentSchema)
class PulseQobjExperiment(QobjExperiment):
    """Model for PulseQobjExperiment inherit from QobjExperiment.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``PulseQobjExperimentSchema``.

    Attributes:
        instructions (list[PulseQobjInstruction]): list of instructions.
    """
    def __init__(self, instructions, **kwargs):

        super().__init__(instructions=instructions,
                         **kwargs)


@bind_schema(PulseQobjConfigSchema)
class PulseQobjConfig(QobjConfig):
    """Model for PulseQobjConfig inherit from QobjConfig.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``PulseQobjConfigSchema``.

    Attributes:
        meas_level (int): a value represents the level of measurement.
        meas_lo_freq (list[float]): local oscillator frequency of measurement pulse.
        meas_return (str): a level of measurement information.
        pulse_library (list[qiskit.qobj.PulseLibraryItem]): a pulse library.
        qubit_lo_freq (list[float]): local oscillator frequency of driving pulse.
    """
    def __init__(self, meas_level, meas_return, pulse_library,
                 qubit_lo_freq, meas_lo_freq, **kwargs):
        self.meas_level = meas_level
        self.meas_return = meas_return
        self.pulse_library = pulse_library
        self.qubit_lo_freq = qubit_lo_freq
        self.meas_lo_freq = meas_lo_freq

        super().__init__(meas_level=meas_level,
                         meas_return=meas_return,
                         pulse_library=pulse_library,
                         qubit_lo_freq=qubit_lo_freq,
                         meas_lo_freq=meas_lo_freq,
                         **kwargs)
