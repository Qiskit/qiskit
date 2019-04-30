# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper class used to convert a pulse instruction into PulseQobjInstruction."""

from qiskit.pulse import commands
from qiskit.pulse.exceptions import PulseError
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


class PulseQobjConverter:
    """
    This class exists for separating entity of pulse instructions and qobj instruction,
    and provides some alleviation of the complexity of the assembler.

    Converter is constructed with qobj model and some experimental configuration,
    and returns proper qobj instruction to each backend.

    Pulse instruction and its qobj are strongly depends on the design of backend,
    and third party providers can be easily add their custom pulse instruction by
    providing custom converter inherit from this.


    To create custom converter for custom instruction
    ```
    class CustomConverter(PulseQobjConverter):

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
    ```
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
            'label': instruction.name,
            'type': instruction.type
        }
        return self._qobj_model(**command_dict)
