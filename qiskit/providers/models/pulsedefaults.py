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

"""Model and schema for pulse defaults."""

from marshmallow.validate import Length, Range

from qiskit.validation import BaseModel, BaseSchema, bind_schema
from qiskit.validation.base import ObjSchema
from qiskit.validation.fields import (Integer, List, Nested, Number, String)
from qiskit.qobj import PulseLibraryItemSchema, PulseQobjInstructionSchema
from qiskit.pulse import CmdDef


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


class CommandSchema(BaseSchema):
    """Schema for Command."""

    # Required properties.
    name = String(required=True)

    # Optional properties.
    qubits = List(Integer(validate=Range(min=0)),
                  validate=Length(min=1))
    sequence = Nested(PulseQobjInstructionSchema, many=True)


class PulseDefaultsSchema(BaseSchema):
    """Schema for PulseDefaults."""

    # Required properties.
    qubit_freq_est = List(Number(), required=True, validate=Length(min=1))
    meas_freq_est = List(Number(), required=True, validate=Length(min=1))
    buffer = Integer(required=True, validate=Range(min=0))
    pulse_library = Nested(PulseLibraryItemSchema, required=True, many=True)
    cmd_def = Nested(CommandSchema, many=True, required=True)

    # Optional properties.
    meas_kernel = Nested(MeasurementKernelSchema)
    discriminator = Nested(DiscriminatorSchema)


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


@bind_schema(CommandSchema)
class Command(BaseModel):
    """Model for Command.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``CommandSchema``.

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
        cmd_def (list[Command]): Backend command definition.
    """

    def __init__(self, qubit_freq_est, meas_freq_est, buffer,
                 pulse_library, cmd_def, **kwargs):
        self.qubit_freq_est = qubit_freq_est
        self.meas_freq_est = meas_freq_est
        self.buffer = buffer
        self.pulse_library = pulse_library
        self.cmd_def = cmd_def

        super().__init__(**kwargs)

    def build_cmd_def(self):
        """Construct the `CmdDef` object for the backend.

        Returns:
            CmdDef: `CmdDef` instance generated from defaults
        """
        return CmdDef.from_defaults(self.cmd_def, self.pulse_library)
