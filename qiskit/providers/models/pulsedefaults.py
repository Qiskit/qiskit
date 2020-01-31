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
import warnings

from typing import Any, Dict, List
from marshmallow.validate import Length, Range

from qiskit.validation import BaseModel, BaseSchema, bind_schema, fields
from qiskit.validation.base import ObjSchema
from qiskit.qobj import PulseLibraryItemSchema, PulseQobjInstructionSchema, PulseLibraryItem
from qiskit.qobj.converters import QobjToInstructionConverter
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.schedule import ParameterizedSchedule


class MeasurementKernelSchema(BaseSchema):
    """Schema for MeasurementKernel."""

    # Optional properties.
    name = fields.String()
    params = fields.Nested(ObjSchema)


class DiscriminatorSchema(BaseSchema):
    """Schema for Discriminator."""

    # Optional properties.
    name = fields.String()
    params = fields.Nested(ObjSchema)


class CommandSchema(BaseSchema):
    """Schema for Command."""

    # Required properties.
    name = fields.String(required=True)

    # Optional properties.
    qubits = fields.List(fields.Integer(validate=Range(min=0)),
                         validate=Length(min=1))
    sequence = fields.Nested(PulseQobjInstructionSchema, many=True)


class PulseDefaultsSchema(BaseSchema):
    """Schema for PulseDefaults."""

    # Required properties.
    qubit_freq_est = fields.List(fields.Number(), required=True, validate=Length(min=1))
    meas_freq_est = fields.List(fields.Number(), required=True, validate=Length(min=1))
    buffer = fields.Integer(required=True, validate=Range(min=0))
    pulse_library = fields.Nested(PulseLibraryItemSchema, required=True, many=True)
    cmd_def = fields.Nested(CommandSchema, many=True, required=True)

    # Optional properties.
    meas_kernel = fields.Nested(MeasurementKernelSchema)
    discriminator = fields.Nested(DiscriminatorSchema)


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
        name: Pulse command name.
    """
    def __init__(self, name: str, **kwargs):
        self.name = name

        super().__init__(**kwargs)


@bind_schema(PulseDefaultsSchema)
class PulseDefaults(BaseModel):
    """Description of default settings for Pulse systems. These are instructions or settings that
    may be good starting points for the Pulse user. The user may modify these defaults for custom
    scheduling.
    """

    _freq_warning_done = False

    def __init__(self,
                 qubit_freq_est: List[float],
                 meas_freq_est: List[float],
                 buffer: int,
                 pulse_library: List[PulseLibraryItem],
                 cmd_def: List[Command],
                 **kwargs: Dict[str, Any]):
        """
        Validate and reformat transport layer inputs to initialize.

        Args:
            qubit_freq_est: Estimated qubit frequencies in GHz.
            meas_freq_est: Estimated measurement cavity frequencies in GHz.
            buffer: Default buffer time (in units of dt) between pulses.
            pulse_library: Pulse name and sample definitions.
            cmd_def: Operation name and definition in terms of Commands.
            **kwargs: Other attributes for the super class.
        """
        super().__init__(**kwargs)

        self.buffer = buffer
        self._qubit_freq_est = [freq * 1e9 for freq in qubit_freq_est]
        self._meas_freq_est = [freq * 1e9 for freq in meas_freq_est]
        self.pulse_library = pulse_library
        self.cmd_def = cmd_def
        self.instruction_schedule_map = InstructionScheduleMap()

        self.converter = QobjToInstructionConverter(pulse_library)
        for inst in cmd_def:
            pulse_insts = [self.converter(inst) for inst in inst.sequence]
            schedule = ParameterizedSchedule(*pulse_insts, name=inst.name)
            self.instruction_schedule_map.add(inst.name, inst.qubits, schedule)

    @property
    def qubit_freq_est(self) -> float:
        """Qubit frequencies in Hertz(Hz)."""
        # only raise qubit_freq_est warning once
        if not PulseDefaults._freq_warning_done:
            warnings.warn('`qubit_freq_est` and `meas_freq_est` now have units of '
                          'Hertz(Hz) rather than gigahertz(GHz).')
            PulseDefaults._freq_warning_done = True

        return self._qubit_freq_est

    @property
    def meas_freq_est(self) -> float:  # pylint: disable=invalid-name
        """Measurement frequencies in Hertz(Hz)."""
        # only raise qubit_freq_est warning once
        if not PulseDefaults._freq_warning_done:
            warnings.warn('`qubit_freq_est` and `meas_freq_est` now have units of '
                          'Hertz(Hz) rather than gigahertz(GHz).')
            PulseDefaults._freq_warning_done = True

        return self._meas_freq_est

    @property
    def circuit_instruction_map(self):
        """Deprecated property, use ``instruction_schedule_map`` instead."""
        warnings.warn("The `circuit_instruction_map` attribute has been renamed to "
                      "`instruction_schedule_map`.", DeprecationWarning)
        return self.instruction_schedule_map

    def __str__(self):
        qubit_freqs = [freq / 1e9 for freq in self.qubit_freq_est]
        meas_freqs = [freq / 1e9 for freq in self.meas_freq_est]
        qfreq = "Qubit Frequencies [GHz]\n{freqs}".format(freqs=qubit_freqs)
        mfreq = "Measurement Frequencies [GHz]\n{freqs} ".format(freqs=meas_freqs)
        return ("<{name}({insts}{qfreq}\n{mfreq})>"
                "".format(name=self.__class__.__name__, insts=str(self.instruction_schedule_map),
                          qfreq=qfreq, mfreq=mfreq))

    def build_cmd_def(self) -> InstructionScheduleMap:
        """
        Return the InstructionScheduleMap built for this PulseDefaults instance.

        Returns:
            InstructionScheduleMap: Generated from defaults.
        """
        warnings.warn("This method is deprecated. Returning a InstructionScheduleMap instead. "
                      "This can be accessed simply through the `instruction_schedule_map` "
                      "attribute of this PulseDefaults instance.", DeprecationWarning)
        return self.instruction_schedule_map
