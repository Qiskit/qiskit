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

# pylint: disable=invalid-name

"""Helper class used to convert a pulse instruction into PulseQobjInstruction."""

import hashlib
import re
import warnings

from enum import Enum
from typing import Union
import numpy as np

from qiskit.pulse import channels, instructions, library
from qiskit.pulse.configuration import Kernel, Discriminator
from qiskit.pulse.exceptions import QiskitError
from qiskit.pulse.parser import parse_string_expr
from qiskit.pulse.schedule import Schedule
from qiskit.qobj import QobjMeasurementOption
from qiskit.qobj.utils import MeasLevel
from qiskit.circuit import Parameter, ParameterExpression

GIGAHERTZ_TO_SI_UNITS = 1e9


class ParametricPulseShapes(Enum):
    """Map the assembled pulse names to the pulse module waveforms.

    The enum name is the transport layer name for pulse shapes, the
    value is its mapping to the OpenPulse Command in Qiskit.
    """

    gaussian = "Gaussian"
    gaussian_square = "GaussianSquare"
    drag = "Drag"
    constant = "Constant"

    @classmethod
    def from_instance(
        cls,
        instance: Union[library.ParametricPulse, library.SymbolicPulse],
    ) -> "ParametricPulseShapes":
        """Get Qobj name from the pulse class instance.

        Args:
            instance: Symbolic or ParametricPulse class.

        Returns:
            Qobj name.

        Raises:
            QiskitError: When pulse instance is not recognizable type.
        """
        if isinstance(instance, library.SymbolicPulse):
            return cls(instance.pulse_type)
        if isinstance(instance, library.parametric_pulses.Gaussian):
            return ParametricPulseShapes.gaussian
        if isinstance(instance, library.parametric_pulses.GaussianSquare):
            return ParametricPulseShapes.gaussian_square
        if isinstance(instance, library.parametric_pulses.Drag):
            return ParametricPulseShapes.drag
        if isinstance(instance, library.parametric_pulses.Constant):
            return ParametricPulseShapes.constant

        raise QiskitError(f"'{instance}' is not valid pulse type.")

    @classmethod
    def to_type(cls, name: str) -> library.SymbolicPulse:
        """Get symbolic pulse class from the name.

        Args:
            name: Qobj name of the pulse.

        Returns:
            Corresponding class.
        """
        return getattr(library, cls[name].value)


class ConversionMethodBinder:
    """Conversion method registrar."""

    def __init__(self):
        """Acts as method registration decorator and tracker for conversion methods."""
        self._bound_instructions = {}

    def __call__(self, bound):
        """Converter decorator method.

        Converter is defined for object to be converted matched on hash

        Args:
            bound (Hashable): Hashable object to bind to the converter.

        """
        # pylint: disable=missing-return-doc, missing-return-type-doc

        def _apply_converter(converter):
            """Return decorated converter function."""
            # Track conversion methods for class.
            self._bound_instructions[bound] = converter
            return converter

        return _apply_converter

    def get_bound_method(self, bound):
        """Get conversion method for bound object."""
        try:
            return self._bound_instructions[bound]
        except KeyError as ex:
            raise QiskitError(f"Bound method for {bound} is not found.") from ex


class InstructionToQobjConverter:
    """Converts pulse Instructions to Qobj models.

    Converter is constructed with qobj model and experimental configuration,
    and returns proper qobj instruction to each backend.

    Third party providers can be add their own custom pulse instructions by
    providing custom converter methods.


    To create a custom converter for custom instruction::

        class CustomConverter(InstructionToQobjConverter):

            @bind_instruction(CustomInstruction)
            def convert_custom_command(self, shift, instruction):
                command_dict = {
                    'name': 'custom_command',
                    't0': shift + instruction.start_time,
                    'param1': instruction.param1,
                    'param2': instruction.param2
                }
                if self._run_config('option1', True):
                    command_dict.update({
                        'param3': instruction.param3
                    })
                return self.qobj_model(**command_dict)
    """

    # class level tracking of conversion methods
    bind_instruction = ConversionMethodBinder()

    def __init__(self, qobj_model, **run_config):
        """Create new converter.

        Args:
             qobj_model (QobjInstruction): marshmallow model to serialize to object.
             run_config (dict): experimental configuration.
        """
        self._qobj_model = qobj_model
        self._run_config = run_config

    def __call__(self, shift, instruction):

        method = self.bind_instruction.get_bound_method(type(instruction))
        return method(self, shift, instruction)

    @bind_instruction(instructions.Acquire)
    def convert_acquire(self, shift, instruction):
        """Return converted `Acquire`.

        Args:
            shift(int): Offset time.
            instruction (Acquire): acquire instruction.
        Returns:
            dict: Dictionary of required parameters.
        """
        meas_level = self._run_config.get("meas_level", 2)
        mem_slot = []
        if instruction.mem_slot:
            mem_slot = [instruction.mem_slot.index]

        command_dict = {
            "name": "acquire",
            "t0": shift + instruction.start_time,
            "duration": instruction.duration,
            "qubits": [instruction.channel.index],
            "memory_slot": mem_slot,
        }
        if meas_level == MeasLevel.CLASSIFIED:
            # setup discriminators
            if instruction.discriminator:
                command_dict.update(
                    {
                        "discriminators": [
                            QobjMeasurementOption(
                                name=instruction.discriminator.name,
                                params=instruction.discriminator.params,
                            )
                        ]
                    }
                )
            # setup register_slots
            if instruction.reg_slot:
                command_dict.update({"register_slot": [instruction.reg_slot.index]})
        if meas_level in [MeasLevel.KERNELED, MeasLevel.CLASSIFIED]:
            # setup kernels
            if instruction.kernel:
                command_dict.update(
                    {
                        "kernels": [
                            QobjMeasurementOption(
                                name=instruction.kernel.name, params=instruction.kernel.params
                            )
                        ]
                    }
                )
        return self._qobj_model(**command_dict)

    def convert_bundled_acquires(
        self,
        shift,
        instructions_,
    ):
        """Bundle a list of acquires instructions at the same time into a single
        Qobj acquire instruction.

        Args:
            shift (int): Offset time.
            instructions_ (List[Acquire]): List of acquire instructions to bundle.
        Returns:
            dict: Dictionary of required parameters.
        Raises:
            QiskitError: If ``instructions`` is empty.
        """
        if not instructions_:
            raise QiskitError('"instructions" may not be empty.')

        meas_level = self._run_config.get("meas_level", 2)

        t0 = instructions_[0].start_time
        duration = instructions_[0].duration
        memory_slots = []
        register_slots = []
        qubits = []
        discriminators = []
        kernels = []

        for instruction in instructions_:
            qubits.append(instruction.channel.index)

            if instruction.start_time != t0:
                raise QiskitError(
                    "The supplied acquire instructions have different starting times. "
                    "Something has gone wrong calling this code. Please report this "
                    "issue."
                )

            if instruction.duration != duration:
                raise QiskitError(
                    "Acquire instructions beginning at the same time must have same duration."
                )

            if instruction.mem_slot:
                memory_slots.append(instruction.mem_slot.index)

            if meas_level == MeasLevel.CLASSIFIED:
                # setup discriminators
                if instruction.discriminator:
                    discriminators.append(
                        QobjMeasurementOption(
                            name=instruction.discriminator.name,
                            params=instruction.discriminator.params,
                        )
                    )
                # setup register_slots
                if instruction.reg_slot:
                    register_slots.append(instruction.reg_slot.index)

            if meas_level in [MeasLevel.KERNELED, MeasLevel.CLASSIFIED]:
                # setup kernels
                if instruction.kernel:
                    kernels.append(
                        QobjMeasurementOption(
                            name=instruction.kernel.name, params=instruction.kernel.params
                        )
                    )
        command_dict = {
            "name": "acquire",
            "t0": t0 + shift,
            "duration": duration,
            "qubits": qubits,
        }
        if memory_slots:
            command_dict["memory_slot"] = memory_slots

        if register_slots:
            command_dict["register_slot"] = register_slots

        if discriminators:
            num_discriminators = len(discriminators)
            if num_discriminators == len(qubits) or num_discriminators == 1:
                command_dict["discriminators"] = discriminators
            else:
                raise QiskitError(
                    "A discriminator must be supplied for every acquisition or a single "
                    "discriminator for all acquisitions."
                )

        if kernels:
            num_kernels = len(kernels)
            if num_kernels == len(qubits) or num_kernels == 1:
                command_dict["kernels"] = kernels
            else:
                raise QiskitError(
                    "A kernel must be supplied for every acquisition or a single "
                    "kernel for all acquisitions."
                )

        return self._qobj_model(**command_dict)

    @bind_instruction(instructions.SetFrequency)
    def convert_set_frequency(self, shift, instruction):
        """Return converted `SetFrequencyInstruction`.

        Args:
            shift (int): Offset time.
            instruction (SetFrequency): set frequency instruction.

        Returns:
            dict: Dictionary of required parameters.
        """
        command_dict = {
            "name": "setf",
            "t0": shift + instruction.start_time,
            "ch": instruction.channel.name,
            "frequency": instruction.frequency / 1e9,
        }
        return self._qobj_model(**command_dict)

    @bind_instruction(instructions.ShiftFrequency)
    def convert_shift_frequency(self, shift, instruction):
        """Return converted `ShiftFrequency`.

        Args:
            shift (int): Offset time.
            instruction (ShiftFrequency): Shift frequency instruction.

        Returns:
            dict: Dictionary of required parameters.
        """
        command_dict = {
            "name": "shiftf",
            "t0": shift + instruction.start_time,
            "ch": instruction.channel.name,
            "frequency": instruction.frequency / 1e9,
        }
        return self._qobj_model(**command_dict)

    @bind_instruction(instructions.SetPhase)
    def convert_set_phase(self, shift, instruction):
        """Return converted `SetPhase`.

        Args:
            shift(int): Offset time.
            instruction (SetPhase): Set phase instruction.
        Returns:
            dict: Dictionary of required parameters.
        """
        command_dict = {
            "name": "setp",
            "t0": shift + instruction.start_time,
            "ch": instruction.channel.name,
            "phase": instruction.phase,
        }
        return self._qobj_model(**command_dict)

    @bind_instruction(instructions.ShiftPhase)
    def convert_shift_phase(self, shift, instruction):
        """Return converted `ShiftPhase`.

        Args:
            shift(int): Offset time.
            instruction (ShiftPhase): Shift phase instruction.
        Returns:
            dict: Dictionary of required parameters.
        """
        command_dict = {
            "name": "fc",
            "t0": shift + instruction.start_time,
            "ch": instruction.channel.name,
            "phase": instruction.phase,
        }
        return self._qobj_model(**command_dict)

    @bind_instruction(instructions.Delay)
    def convert_delay(self, shift, instruction):
        """Return converted `Delay`.

        Args:
            shift(int): Offset time.
            instruction (Delay): Delay instruction.
        Returns:
            dict: Dictionary of required parameters.
        """
        command_dict = {
            "name": "delay",
            "t0": shift + instruction.start_time,
            "ch": instruction.channel.name,
            "duration": instruction.duration,
        }
        return self._qobj_model(**command_dict)

    @bind_instruction(instructions.Play)
    def convert_play(self, shift, instruction):
        """Return the converted `Play`.

        Args:
            shift (int): Offset time.
            instruction (Play): An instance of Play.
        Returns:
            dict: Dictionary of required parameters.
        """
        if isinstance(instruction.pulse, (library.ParametricPulse, library.SymbolicPulse)):
            params = dict(instruction.pulse.parameters)
            # IBM backends expect "amp" to be the complex amplitude
            if "amp" in params and "angle" in params:
                params["amp"] = complex(params["amp"] * np.exp(1j * params["angle"]))
                del params["angle"]

            command_dict = {
                "name": "parametric_pulse",
                "pulse_shape": ParametricPulseShapes.from_instance(instruction.pulse).name,
                "t0": shift + instruction.start_time,
                "ch": instruction.channel.name,
                "parameters": params,
            }
        else:
            command_dict = {
                "name": instruction.name,
                "t0": shift + instruction.start_time,
                "ch": instruction.channel.name,
            }

        return self._qobj_model(**command_dict)

    @bind_instruction(instructions.Snapshot)
    def convert_snapshot(self, shift, instruction):
        """Return converted `Snapshot`.

        Args:
            shift(int): Offset time.
            instruction (Snapshot): snapshot instruction.
        Returns:
            dict: Dictionary of required parameters.
        """
        command_dict = {
            "name": "snapshot",
            "t0": shift + instruction.start_time,
            "label": instruction.label,
            "type": instruction.type,
        }
        return self._qobj_model(**command_dict)


class QobjToInstructionConverter:
    """Converts Qobj models to pulse Instructions"""

    # class level tracking of conversion methods
    bind_name = ConversionMethodBinder()
    chan_regex = re.compile(r"([a-zA-Z]+)(\d+)")

    def __init__(self, pulse_library, **run_config):
        """Create new converter.

        Args:
             pulse_library (List[PulseLibraryItem]): Pulse library to be used in conversion
             run_config (dict): experimental configuration.
        """
        self._run_config = run_config
        # bind pulses to conversion methods
        for pulse in pulse_library:
            self.bind_pulse(pulse)

    def __call__(self, instruction):
        method = self.bind_name.get_bound_method(instruction.name)
        return method(self, instruction)

    def get_channel(self, channel: str) -> channels.PulseChannel:
        """Parse and retrieve channel from ch string.

        Args:
            channel: Channel to match

        Returns:
            Matched channel

        Raises:
            QiskitError: Is raised if valid channel is not matched
        """
        match = self.chan_regex.match(channel)
        if match:
            prefix, index = match.group(1), int(match.group(2))

            if prefix == channels.DriveChannel.prefix:
                return channels.DriveChannel(index)
            elif prefix == channels.MeasureChannel.prefix:
                return channels.MeasureChannel(index)
            elif prefix == channels.ControlChannel.prefix:
                return channels.ControlChannel(index)

        raise QiskitError("Channel %s is not valid" % channel)

    @staticmethod
    def disassemble_value(value_expr: Union[float, str]) -> Union[float, ParameterExpression]:
        """A helper function to format instruction operand.

        If parameter in string representation is specified, this method parses the
        input string and generates Qiskit ParameterExpression object.

        Args:
            value_expr: Operand value in Qobj.

        Returns:
            Parsed operand value. ParameterExpression object is returned if value is not number.
        """
        if isinstance(value_expr, str):
            str_expr = parse_string_expr(value_expr, partial_binding=False)
            value_expr = str_expr(**{pname: Parameter(pname) for pname in str_expr.params})
        return value_expr

    @bind_name("acquire")
    def convert_acquire(self, instruction):
        """Return converted `Acquire`.

        Args:
            instruction (PulseQobjInstruction): acquire qobj
        Returns:
            Schedule: Converted and scheduled Instruction
        """
        t0 = instruction.t0
        duration = instruction.duration
        qubits = instruction.qubits
        acquire_channels = [channels.AcquireChannel(qubit) for qubit in qubits]

        mem_slots = [channels.MemorySlot(instruction.memory_slot[i]) for i in range(len(qubits))]

        if hasattr(instruction, "register_slot"):
            register_slots = [
                channels.RegisterSlot(instruction.register_slot[i]) for i in range(len(qubits))
            ]
        else:
            register_slots = [None] * len(qubits)

        discriminators = (
            instruction.discriminators if hasattr(instruction, "discriminators") else None
        )
        if not isinstance(discriminators, list):
            discriminators = [discriminators]
        if any(discriminators[i] != discriminators[0] for i in range(len(discriminators))):
            warnings.warn(
                "Can currently only support one discriminator per acquire. Defaulting "
                "to first discriminator entry."
            )
        discriminator = discriminators[0]
        if discriminator:
            discriminator = Discriminator(name=discriminators[0].name, **discriminators[0].params)

        kernels = instruction.kernels if hasattr(instruction, "kernels") else None
        if not isinstance(kernels, list):
            kernels = [kernels]
        if any(kernels[0] != kernels[i] for i in range(len(kernels))):
            warnings.warn(
                "Can currently only support one kernel per acquire. Defaulting to first "
                "kernel entry."
            )
        kernel = kernels[0]
        if kernel:
            kernel = Kernel(name=kernels[0].name, **kernels[0].params)

        schedule = Schedule()

        for acquire_channel, mem_slot, reg_slot in zip(acquire_channels, mem_slots, register_slots):
            schedule |= (
                instructions.Acquire(
                    duration,
                    acquire_channel,
                    mem_slot=mem_slot,
                    reg_slot=reg_slot,
                    kernel=kernel,
                    discriminator=discriminator,
                )
                << t0
            )

        return schedule

    @bind_name("setp")
    def convert_set_phase(self, instruction):
        """Return converted `SetPhase`.

        Args:
            instruction (PulseQobjInstruction): phase set qobj instruction
        Returns:
            Schedule: Converted and scheduled Instruction
        """
        t0 = instruction.t0
        channel = self.get_channel(instruction.ch)
        phase = self.disassemble_value(instruction.phase)

        return instructions.SetPhase(phase, channel) << t0

    @bind_name("fc")
    def convert_shift_phase(self, instruction):
        """Return converted `ShiftPhase`.

        Args:
            instruction (PulseQobjInstruction): phase shift qobj instruction
        Returns:
            Schedule: Converted and scheduled Instruction
        """
        t0 = instruction.t0
        channel = self.get_channel(instruction.ch)
        phase = self.disassemble_value(instruction.phase)

        return instructions.ShiftPhase(phase, channel) << t0

    @bind_name("setf")
    def convert_set_frequency(self, instruction):
        """Return converted `SetFrequencyInstruction`.

        Args:
            instruction (PulseQobjInstruction): set frequency qobj instruction
                                                The input frequency is expressed  in GHz,
                                                so it will be scaled by a factor 1e9.
        Returns:
            Schedule: Converted and scheduled Instruction
        """
        t0 = instruction.t0
        channel = self.get_channel(instruction.ch)
        frequency = self.disassemble_value(instruction.frequency) * GIGAHERTZ_TO_SI_UNITS

        return instructions.SetFrequency(frequency, channel) << t0

    @bind_name("shiftf")
    def convert_shift_frequency(self, instruction):
        """Return converted `ShiftFrequency`.

        Args:
            instruction (PulseQobjInstruction): Shift frequency qobj instruction.
                                                The input frequency is expressed  in GHz,
                                                so it will be scaled by a factor 1e9.

        Returns:
            Schedule: Converted and scheduled Instruction
        """
        t0 = instruction.t0
        channel = self.get_channel(instruction.ch)
        frequency = self.disassemble_value(instruction.frequency) * GIGAHERTZ_TO_SI_UNITS

        return instructions.ShiftFrequency(frequency, channel) << t0

    @bind_name("delay")
    def convert_delay(self, instruction):
        """Return converted `Delay`.

        Args:
            instruction (Delay): Delay qobj instruction

        Returns:
            Schedule: Converted and scheduled Instruction
        """
        t0 = instruction.t0
        channel = self.get_channel(instruction.ch)
        duration = instruction.duration
        return instructions.Delay(duration, channel) << t0

    def bind_pulse(self, pulse):
        """Bind the supplied pulse to a converter method by pulse name.

        Args:
            pulse (PulseLibraryItem): Pulse to bind
        """
        pulse = library.Waveform(pulse.samples, pulse.name)

        @self.bind_name(pulse.name)
        def convert_named_drive(self, instruction):
            """Return converted `Play`.

            Args:
                instruction (PulseQobjInstruction): pulse qobj
            Returns:
                Schedule: Converted and scheduled pulse
            """
            t0 = instruction.t0
            channel = self.get_channel(instruction.ch)
            return instructions.Play(pulse, channel) << t0

    @bind_name("parametric_pulse")
    def convert_parametric(self, instruction):
        """Return the ParametricPulse implementation that is described by the instruction.

        If parametric pulse label is not provided by the backend, this method naively generates
        a pulse name based on the pulse shape and bound parameters. This pulse name is formatted
        to, for example, `gaussian_a4e3`, here the last four digits are a part of
        the hash string generated based on the pulse shape and the parameters.
        Because we are using a truncated hash for readability,
        there may be a small risk of pulse name collision with other pulses.
        Basically the parametric pulse name is used just for visualization purpose and
        the pulse module should not have dependency on the parametric pulse names.

        Args:
            instruction (PulseQobjInstruction): pulse qobj
        Returns:
            Schedule: Schedule containing the converted pulse
        """
        t0 = instruction.t0
        channel = self.get_channel(instruction.ch)

        try:
            pulse_name = instruction.label
        except AttributeError:
            sorted_params = sorted(tuple(instruction.parameters.items()), key=lambda x: x[0])
            base_str = "{pulse}_{params}".format(
                pulse=instruction.pulse_shape, params=str(sorted_params)
            )
            short_pulse_id = hashlib.md5(base_str.encode("utf-8")).hexdigest()[:4]
            pulse_name = f"{instruction.pulse_shape}_{short_pulse_id}"
        params = dict(instruction.parameters)
        if "amp" in params and isinstance(params["amp"], complex):
            params["angle"] = np.angle(params["amp"])
            params["amp"] = np.abs(params["amp"])

        pulse = ParametricPulseShapes.to_type(instruction.pulse_shape)(**params, name=pulse_name)
        return instructions.Play(pulse, channel) << t0

    @bind_name("snapshot")
    def convert_snapshot(self, instruction):
        """Return converted `Snapshot`.

        Args:
            instruction (PulseQobjInstruction): snapshot qobj
        Returns:
            Schedule: Converted and scheduled Snapshot
        """
        t0 = instruction.t0
        return instructions.Snapshot(instruction.label, instruction.type) << t0
