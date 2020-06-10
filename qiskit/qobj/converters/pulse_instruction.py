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

"""Helper class used to convert a pulse instruction into PulseQobjInstruction."""

import re
import warnings

from enum import Enum

from qiskit.pulse import commands, channels, instructions, pulse_lib
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.configuration import Kernel, Discriminator
from qiskit.pulse.parser import parse_string_expr
from qiskit.pulse.schedule import ParameterizedSchedule, Schedule
from qiskit.qobj import QobjMeasurementOption
from qiskit.qobj.utils import MeasLevel


class ParametricPulseShapes(Enum):
    """Map the assembled pulse names to the pulse module commands.

    The enum name is the transport layer name for pulse shapes, the
    value is its mapping to the OpenPulse Command in Qiskit.
    """
    gaussian = commands.Gaussian
    gaussian_square = commands.GaussianSquare
    drag = commands.Drag
    constant = commands.Constant


class ConversionMethodBinder:
    """Conversion method registrar."""
    def __init__(self):
        """Acts as method registration decorator and tracker for conversion methods."""
        self._bound_instructions = {}

    def __call__(self, bound):
        """ Converter decorator method.

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
        except KeyError:
            raise PulseError('Bound method for %s is not found.' % bound)


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

    @bind_instruction(commands.AcquireInstruction)
    def convert_acquire_deprecated(self, shift, instruction):
        """Return converted `AcquireInstruction`.

        Args:
            shift(int): Offset time.
            instruction (AcquireInstruction): acquire instruction.
        Returns:
            dict: Dictionary of required parameters.
        """
        meas_level = self._run_config.get('meas_level', 2)
        mem_slot = []
        if instruction.mem_slot:
            mem_slot = [instruction.mem_slot.index]

        command_dict = {
            'name': 'acquire',
            't0': shift + instruction.start_time,
            'duration': instruction.duration,
            'qubits': [instruction.acquire.index],
            'memory_slot': mem_slot
        }
        if meas_level == MeasLevel.CLASSIFIED:
            # setup discriminators
            if instruction.command.discriminator:
                command_dict.update({
                    'discriminators': [
                        QobjMeasurementOption(
                            name=instruction.command.discriminator.name,
                            params=instruction.command.discriminator.params)
                    ]
                })
            # setup register_slots
            if instruction.reg_slot:
                command_dict.update({
                    'register_slot': [instruction.reg_slot.index]
                })
        if meas_level in [MeasLevel.KERNELED, MeasLevel.CLASSIFIED]:
            # setup kernels
            if instruction.command.kernel:
                command_dict.update({
                    'kernels': [
                        QobjMeasurementOption(
                            name=instruction.command.kernel.name,
                            params=instruction.command.kernel.params)
                    ]
                })
        return self._qobj_model(**command_dict)

    @bind_instruction(instructions.Acquire)
    def convert_acquire(self, shift, instruction):
        """Return converted `Acquire`.

        Args:
            shift(int): Offset time.
            instruction (Acquire): acquire instruction.
        Returns:
            dict: Dictionary of required parameters.
        """
        meas_level = self._run_config.get('meas_level', 2)
        mem_slot = []
        if instruction.mem_slot:
            mem_slot = [instruction.mem_slot.index]

        command_dict = {
            'name': 'acquire',
            't0': shift + instruction.start_time,
            'duration': instruction.duration,
            'qubits': [instruction.channel.index],
            'memory_slot': mem_slot
        }
        if meas_level == MeasLevel.CLASSIFIED:
            # setup discriminators
            if instruction.discriminator:
                command_dict.update({
                    'discriminators': [
                        QobjMeasurementOption(
                            name=instruction.discriminator.name,
                            params=instruction.discriminator.params)
                    ]
                })
            # setup register_slots
            if instruction.reg_slot:
                command_dict.update({
                    'register_slot': [instruction.reg_slot.index]
                })
        if meas_level in [MeasLevel.KERNELED, MeasLevel.CLASSIFIED]:
            # setup kernels
            if instruction.kernel:
                command_dict.update({
                    'kernels': [
                        QobjMeasurementOption(
                            name=instruction.kernel.name,
                            params=instruction.kernel.params)
                    ]
                })
        return self._qobj_model(**command_dict)

    def convert_single_acquires(self, shift, instruction,
                                qubits=None, memory_slot=None, register_slot=None):
        """Return converted `AcquireInstruction`, with options to override the qubits,
        memory_slot, and register_slot fields. This is useful for grouping
        AcquisitionInstructions which are operated on a single AcquireChannel and
        a single MemorySlot.

        Args:
            shift (int): Offset time.
            instruction (AcquireInstruction): acquire instruction.
            qubits (list(int)): A list of qubit indices to acquire.
            memory_slot (list(int)): A list of memory slot indices to store results.
            register_slot (list(int)): A list of register slot addresses to store results.
        Returns:
            dict: Dictionary of required parameters.
        """
        res = self.convert_acquire(shift, instruction)
        if qubits:
            res.qubits = qubits
        if memory_slot:
            res.memory_slot = memory_slot
        if register_slot:
            res.register_slot = register_slot
        return res

    @bind_instruction(commands.FrameChangeInstruction)
    def convert_frame_change(self, shift, instruction):
        """Return converted `FrameChangeInstruction`.

        Args:
            shift(int): Offset time.
            instruction (FrameChangeInstruction): frame change instruction.
        Returns:
            dict: Dictionary of required parameters.
        """
        command_dict = {
            'name': 'fc',
            't0': shift + instruction.start_time,
            'ch': instruction.channels[0].name,
            'phase': instruction.command.phase
        }
        return self._qobj_model(**command_dict)

    @bind_instruction(instructions.SetFrequency)
    def convert_set_frequency(self, shift, instruction):
        """ Return converted `SetFrequencyInstruction`.

        Args:
            shift (int): Offset time.
            instruction (SetFrequency): set frequency instruction.

        Returns:
            dict: Dictionary of required parameters.
        """
        command_dict = {
            'name': 'setf',
            't0': shift+instruction.start_time,
            'ch': instruction.channel.name,
            'frequency': instruction.frequency / 1e9
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
            'name': 'shiftf',
            't0': shift+instruction.start_time,
            'ch': instruction.channel.name,
            'frequency': instruction.frequency / 1e9
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
            'name': 'fc',
            't0': shift + instruction.start_time,
            'ch': instruction.channel.name,
            'phase': instruction.phase
        }
        return self._qobj_model(**command_dict)

    @bind_instruction(commands.PersistentValueInstruction)
    def convert_persistent_value(self, shift, instruction):
        """Return converted `PersistentValueInstruction`.

        Args:
            shift(int): Offset time.
            instruction (PersistentValueInstruction): persistent value instruction.
        Returns:
            dict: Dictionary of required parameters.
        """
        warnings.warn("The PersistentValue command is deprecated. Use qiskit.pulse.Constant "
                      "instead.", DeprecationWarning)
        command_dict = {
            'name': 'pv',
            't0': shift + instruction.start_time,
            'ch': instruction.channels[0].name,
            'val': instruction.command.value
        }
        return self._qobj_model(**command_dict)

    @bind_instruction(commands.PulseInstruction)
    def convert_drive(self, shift, instruction):
        """Return converted `PulseInstruction`.

        Args:
            shift(int): Offset time.
            instruction (PulseInstruction): drive instruction.
        Returns:
            dict: Dictionary of required parameters.
        """
        command_dict = {
            'name': instruction.command.name,
            't0': shift + instruction.start_time,
            'ch': instruction.channels[0].name
        }
        return self._qobj_model(**command_dict)

    @bind_instruction(commands.ParametricInstruction)
    def convert_parametric(self, shift, instruction):
        """Return the converted `ParametricInstruction`.

        Args:
            shift (int): Offset time.
            instruction (ParametricInstruction): An instance of a ParametricInstruction subclass.
        Returns:
            dict: Dictionary of required parameters.
        """
        command_dict = {
            'name': 'parametric_pulse',
            'pulse_shape': ParametricPulseShapes(type(instruction.command)).name,
            't0': shift + instruction.start_time,
            'ch': instruction.channels[0].name,
            'parameters': instruction.command.parameters
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
        if isinstance(instruction.pulse, pulse_lib.ParametricPulse):
            command_dict = {
                'name': 'parametric_pulse',
                'pulse_shape': ParametricPulseShapes(type(instruction.pulse)).name,
                't0': shift + instruction.start_time,
                'ch': instruction.channel.name,
                'parameters': instruction.pulse.parameters
            }
        else:
            command_dict = {
                'name': instruction.name,
                't0': shift + instruction.start_time,
                'ch': instruction.channel.name
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
            'name': 'snapshot',
            't0': shift + instruction.start_time,
            'label': instruction.label,
            'type': instruction.type
        }
        return self._qobj_model(**command_dict)


class QobjToInstructionConverter:
    """Converts Qobj models to pulse Instructions
    """
    # pylint: disable=invalid-name,missing-return-type-doc
    # class level tracking of conversion methods
    bind_name = ConversionMethodBinder()
    chan_regex = re.compile(r'([a-zA-Z]+)(\d+)')

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

    def get_channel(self, channel):
        """Parse and retrieve channel from ch string.

        Args:
            channel (str): Channel to match

        Returns:
            (Channel, int): Matched channel

        Raises:
            PulseError: Is raised if valid channel is not matched
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

        raise PulseError('Channel %s is not valid' % channel)

    @bind_name('acquire')
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

        if hasattr(instruction, 'register_slot'):
            register_slots = [channels.RegisterSlot(instruction.register_slot[i])
                              for i in range(len(qubits))]
        else:
            register_slots = [None] * len(qubits)

        discriminators = (instruction.discriminators
                          if hasattr(instruction, 'discriminators') else None)
        if not isinstance(discriminators, list):
            discriminators = [discriminators]
        if any(discriminators[i] != discriminators[0] for i in range(len(discriminators))):
            warnings.warn("Can currently only support one discriminator per acquire. Defaulting "
                          "to first discriminator entry.")
        discriminator = discriminators[0]
        if discriminator:
            discriminator = Discriminator(name=discriminators[0].name, **discriminators[0].params)

        kernels = (instruction.kernels
                   if hasattr(instruction, 'kernels') else None)
        if not isinstance(kernels, list):
            kernels = [kernels]
        if any(kernels[0] != kernels[i] for i in range(len(kernels))):
            warnings.warn("Can currently only support one kernel per acquire. Defaulting to first "
                          "kernel entry.")
        kernel = kernels[0]
        if kernel:
            kernel = Kernel(name=kernels[0].name, **kernels[0].params)

        schedule = Schedule()

        for acquire_channel, mem_slot, reg_slot in zip(acquire_channels, mem_slots, register_slots):
            schedule |= instructions.Acquire(duration, acquire_channel, mem_slot=mem_slot,
                                             reg_slot=reg_slot, kernel=kernel,
                                             discriminator=discriminator) << t0

        return schedule

    @bind_name('fc')
    def convert_shift_phase(self, instruction):
        """Return converted `ShiftPhase`.

        Args:
            instruction (PulseQobjInstruction): phase shift qobj instruction
        Returns:
            Schedule: Converted and scheduled Instruction
        """
        t0 = instruction.t0
        channel = self.get_channel(instruction.ch)
        phase = instruction.phase

        # This is parameterized
        if isinstance(phase, str):
            phase_expr = parse_string_expr(phase, partial_binding=False)

            def gen_fc_sched(*args, **kwargs):
                # this should be real value
                _phase = phase_expr(*args, **kwargs)
                return instructions.ShiftPhase(_phase, channel) << t0

            return ParameterizedSchedule(gen_fc_sched, parameters=phase_expr.params)

        return instructions.ShiftPhase(phase, channel) << t0

    @bind_name('setf')
    def convert_set_frequency(self, instruction):
        """Return converted `SetFrequencyInstruction`.

        Args:
            instruction (PulseQobjInstruction): set frequency qobj instruction
        Returns:
            Schedule: Converted and scheduled Instruction
        """
        t0 = instruction.t0
        channel = self.get_channel(instruction.ch)
        frequency = instruction.frequency * 1e9

        if isinstance(frequency, str):
            frequency_expr = parse_string_expr(frequency, partial_binding=False)

            def gen_sf_schedule(*args, **kwargs):
                _frequency = frequency_expr(*args, **kwargs)
                return instructions.SetFrequency(_frequency, channel) << t0

            return ParameterizedSchedule(gen_sf_schedule, parameters=frequency_expr.params)

        return instructions.SetFrequency(frequency, channel) << t0

    @bind_name('shiftf')
    def convert_shift_frequency(self, instruction):
        """Return converted `ShiftFrequency`.

        Args:
            instruction (PulseQobjInstruction): Shift frequency qobj instruction.

        Returns:
            Schedule: Converted and scheduled Instruction
        """
        t0 = instruction.t0
        channel = self.get_channel(instruction.ch)
        frequency = instruction.frequency * 1e9

        if isinstance(frequency, str):
            frequency_expr = parse_string_expr(frequency, partial_binding=False)

            def gen_sf_schedule(*args, **kwargs):
                _frequency = frequency_expr(*args, **kwargs)
                return instructions.ShiftFrequency(_frequency, channel) << t0

            return ParameterizedSchedule(gen_sf_schedule, parameters=frequency_expr.params)

        return instructions.ShiftFrequency(frequency, channel) << t0

    @bind_name('delay')
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

    @bind_name('pv')
    def convert_persistent_value(self, instruction):
        """Return converted `PersistentValueInstruction`.

        Args:
            instruction (PulseQobjInstruction): persistent value qobj
        Returns:
            Schedule: Converted and scheduled Instruction
        """
        t0 = instruction.t0
        channel = self.get_channel(instruction.ch)
        val = instruction.val

        # This is parameterized
        if isinstance(val, str):
            val_expr = parse_string_expr(val, partial_binding=False)

            def gen_pv_sched(*args, **kwargs):
                val = complex(val_expr(*args, **kwargs))
                return commands.PersistentValue(val)(channel) << t0

            return ParameterizedSchedule(gen_pv_sched, parameters=val_expr.params)

        return commands.PersistentValue(val)(channel) << t0

    def bind_pulse(self, pulse):
        """Bind the supplied pulse to a converter method by pulse name.

        Args:
            pulse (PulseLibraryItem): Pulse to bind
        """
        # pylint: disable=unused-variable
        pulse = pulse_lib.SamplePulse(pulse.samples, pulse.name)

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

    @bind_name('parametric_pulse')
    def convert_parametric(self, instruction):
        """Return the ParametricPulse implementation that is described by the instruction.

        Args:
            instruction (PulseQobjInstruction): pulse qobj
        Returns:
            Schedule: Schedule containing the converted pulse
        """
        t0 = instruction.t0
        channel = self.get_channel(instruction.ch)
        pulse = ParametricPulseShapes[instruction.pulse_shape].value(**instruction.parameters)
        return instructions.Play(pulse, channel) << t0

    @bind_name('snapshot')
    def convert_snapshot(self, instruction):
        """Return converted `Snapshot`.

        Args:
            instruction (PulseQobjInstruction): snapshot qobj
        Returns:
            Schedule: Converted and scheduled Snapshot
        """
        t0 = instruction.t0
        return instructions.Snapshot(instruction.label, instruction.type) << t0
