# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Model and schema for pulse defaults."""

from marshmallow.validate import Length, Range

from qiskit.validation import BaseModel, BaseSchema, bind_schema
from qiskit.validation.base import ObjSchema
from qiskit.validation.fields import (Complex, Integer, List, Nested, Number,
                                      String)


class PulseLibraryItemSchema(BaseSchema):
    """Schema for PulseLibraryItem."""

    # Required properties.
    name = String(required=True)
    samples = List(Complex(), required=True,
                   validate=Length(min=1))


class MeasurementKernelSchema(BaseSchema):
    """Schema for MeasurementKernel."""

    # Optional properties.
    name = String()
    params = Nested(ObjSchema)


class DiscriminatorSchema(BaseSchema):
    """Schema for Discriminator."""

    # Optional properties.
    name = String()
    params = Nested(ObjSchema)


class PulseCommandSchema(BaseSchema):
    """Schema for PulseCommand."""

    # Required properties.
    name = String(required=True)

    # Optional properties.
    qubits = List(Integer(validate=Range(min=0)),
                  validate=Length(min=1))
    sequence = Nested(ObjSchema, many=True)


class PulseDefaultsSchema(BaseSchema):
    """Schema for PulseDefaults."""

    # Required properties.
    qubit_freq_est = List(Number(), required=True, validate=Length(min=1))
    meas_freq_est = List(Number(), required=True, validate=Length(min=1))
    buffer = Integer(required=True, validate=Range(min=0))
    pulse_library = Nested(PulseLibraryItemSchema, required=True, many=True)
    cmd_def = Nested(PulseCommandSchema, many=True, required=True)

    # Optional properties.
    meas_kernel = Nested(MeasurementKernelSchema)
    discriminator = Nested(DiscriminatorSchema)


@bind_schema(PulseLibraryItemSchema)
class PulseLibraryItem(BaseModel):
    """Model for PulseLibraryItem.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``PulseLibraryItemSchema``.

    Attributes:
        name (str): Pulse name.
        samples (list[complex]): Pulse samples.
    """
    def __init__(self, name, samples, **kwargs):
        self.name = name
        self.samples = samples

        super().__init__(**kwargs)


@bind_schema(MeasurementKernelSchema)
class MeasurementKernel(BaseModel):
    """Model for MeasurementKernel.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``MeasurementKernelSchema``.
    """
    pass


@bind_schema(DiscriminatorSchema)
class Discriminator(BaseModel):
    """Model for Discriminator.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``DiscriminatorSchema``.
    """
    pass


@bind_schema(PulseCommandSchema)
class PulseCommand(BaseModel):
    """Model for PulseCommand.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``PulseCommandSchema``.

    Attributes:
        name (str): Pulse command name.
    """
    def __init__(self, name, **kwargs):
        self.name = name

        super().__init__(**kwargs)


@bind_schema(PulseDefaultsSchema)
class PulseDefaults(BaseModel):
    """Model for PulseDefaults.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``PulseDefaultsSchema``.

    Attributes:
        qubit_freq_est (list[number]): Estimated qubit frequencies in GHz.
        meas_freq_est (list[number]): Estimated measurement cavity frequencies
            in GHz.
        buffer (int): Default buffer time (in units of dt) between pulses.
        pulse_library (list[PulseLibraryItem]): Backend pulse library.
        cmd_def (list[PulseCommand]): Backend command definition.
    """

    def __init__(self, qubit_freq_est, meas_freq_est, buffer,
                 pulse_library, cmd_def, **kwargs):
        self.qubit_freq_est = qubit_freq_est
        self.meas_freq_est = meas_freq_est
        self.buffer = buffer
        self.pulse_library = pulse_library
        self.cmd_def = cmd_def

        super().__init__(**kwargs)
