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

from qiskit.pulse import commands, channels
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.parser import parse_string_expr
from qiskit.pulse.schedule import ParameterizedSchedule, Schedule
from qiskit.qobj import QobjMeasurementOption


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
                    't0': shift+instruction.start_time,
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
    def convert_acquire(self, shift, instruction):
        """Return converted `AcquireInstruction`.

        Args:
            shift(int): Offset time.
            instruction (AcquireInstruction): acquire instruction.
        Returns:
            dict: Dictionary of required parameters.
        """
        meas_level = self._run_config.get('meas_level', 2)

        command_dict = {
            'name': 'acquire',
            't0': shift+instruction.start_time,
            'duration': instruction.duration,
            'qubits': [q.index for q in instruction.acquires],
            'memory_slot': [m.index for m in instruction.mem_slots]
        }
        if meas_level == 2:
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
            if instruction.reg_slots:
                command_dict.update({
                    'register_slot': [regs.index for regs in instruction.reg_slots]
                })
        if meas_level >= 1:
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
            't0': shift+instruction.start_time,
            'ch': instruction.channels[0].name,
            'phase': instruction.command.phase
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
        command_dict = {
            'name': 'pv',
            't0': shift+instruction.start_time,
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
            't0': shift+instruction.start_time,
            'ch': instruction.channels[0].name
        }
        return self._qobj_model(**command_dict)

    @bind_instruction(commands.Snapshot)
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
            't0': shift+instruction.start_time,
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

    def __init__(self, pulse_library, buffer=0, **run_config):
        """Create new converter.

        Args:
             pulse_library (List[PulseLibraryItem]): Pulse library to be used in conversion
             buffer (int): Channel buffer
             run_config (dict): experimental configuration.
        """
        self.buffer = buffer
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
                return channels.DriveChannel(index, buffer=self.buffer)
            elif prefix == channels.MeasureChannel.prefix:
                return channels.MeasureChannel(index, buffer=self.buffer)
            elif prefix == channels.ControlChannel.prefix:
                return channels.ControlChannel(index, buffer=self.buffer)

        raise PulseError('Channel %s is not valid' % channel)

    @bind_name('acquire')
    def convert_acquire(self, instruction):
        """Return converted `AcquireInstruction`.

        Args:
            instruction (PulseQobjInstruction): acquire qobj
        Returns:
            Schedule: Converted and scheduled Instruction
        """
        t0 = instruction.t0
        duration = instruction.duration
        qubits = instruction.qubits
        qubit_channels = [channels.AcquireChannel(qubit, buffer=self.buffer) for qubit in qubits]

        mem_slots = [channels.MemorySlot(instruction.memory_slot[i]) for i in range(len(qubits))]

        if hasattr(instruction, 'register_slot'):
            register_slots = [channels.RegisterSlot(instruction.register_slot[i])
                              for i in range(len(qubits))]
        else:
            register_slots = None

        discriminators = (instruction.discriminators
                          if hasattr(instruction, 'discriminators') else None)
        if not isinstance(discriminators, list):
            discriminators = [discriminators]
        if any(discriminators[i] != discriminators[0] for i in range(len(discriminators))):
            warnings.warn("Can currently only support one discriminator per acquire. Defaulting "
                          "to first discriminator entry.")
        discriminator = discriminators[0]
        if discriminator:
            discriminator = commands.Discriminator(name=discriminators[0].name,
                                                   params=discriminators[0].params)

        kernels = (instruction.kernels
                   if hasattr(instruction, 'kernels') else None)
        if not isinstance(kernels, list):
            kernels = [kernels]
        if any(kernels[0] != kernels[i] for i in range(len(kernels))):
            warnings.warn("Can currently only support one kernel per acquire. Defaulting to first "
                          "kernel entry.")
        kernel = kernels[0]
        if kernel:
            kernel = commands.Kernel(name=kernels[0].name, params=kernels[0].params)

        cmd = commands.Acquire(duration, discriminator=discriminator, kernel=kernel)
        schedule = Schedule()
        schedule |= commands.AcquireInstruction(cmd, qubit_channels, mem_slots,
                                                register_slots) << t0

        return schedule

    @bind_name('fc')
    def convert_frame_change(self, instruction):
        """Return converted `FrameChangeInstruction`.

        Args:
            instruction (PulseQobjInstruction): frame change qobj
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
                return commands.FrameChange(_phase)(channel) << t0

            return ParameterizedSchedule(gen_fc_sched, parameters=phase_expr.params)

        return commands.FrameChange(phase)(channel) << t0

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
        pulse = commands.SamplePulse(pulse.samples, pulse.name)

        @self.bind_name(pulse.name)
        def convert_named_drive(self, instruction):
            """Return converted `PulseInstruction`.

            Args:
                instruction (PulseQobjInstruction): pulse qobj
            Returns:
                Schedule: Converted and scheduled pulse
            """
            t0 = instruction.t0
            channel = self.get_channel(instruction.ch)
            return pulse(channel) << t0

    @bind_name('snapshot')
    def convert_snapshot(self, instruction):
        """Return converted `Snapshot`.

        Args:
            instruction (PulseQobjInstruction): snapshot qobj
        Returns:
            Schedule: Converted and scheduled Snapshot
        """
        t0 = instruction.t0
        return commands.Snapshot(instruction.label, instruction.type) << t0
