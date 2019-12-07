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

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Set, Tuple, Union
from marshmallow.validate import Length, Range

from qiskit.validation import BaseModel, BaseSchema, bind_schema, fields
from qiskit.validation.base import ObjSchema
from qiskit.qobj import PulseLibraryItemSchema, PulseQobjInstructionSchema, PulseLibraryItem
from qiskit.qobj.converters import QobjToInstructionConverter
from qiskit.pulse.schedule import Schedule, ParameterizedSchedule
from qiskit.pulse.exceptions import PulseError


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
        self.qubit_freq_est = [freq * 1e9 for freq in qubit_freq_est]
        self.meas_freq_est = [freq * 1e9 for freq in meas_freq_est]
        self.pulse_library = pulse_library
        self.cmd_def = cmd_def
        self.instruction_schedules = InstructionScheduleMap()

        self.converter = QobjToInstructionConverter(pulse_library)
        for inst in cmd_def:
            pulse_insts = [self.converter(inst) for inst in inst.sequence]
            schedule = ParameterizedSchedule(*pulse_insts, name=inst.name)
            self.instruction_schedules.add(inst.name, inst.qubits, schedule)

    def __str__(self):
        qubit_freqs = [freq / 1e9 for freq in self.qubit_freq_est]
        meas_freqs = [freq / 1e9 for freq in self.meas_freq_est]
        qfreq = "Qubit Frequencies [GHz]\n{freqs}".format(freqs=qubit_freqs)
        mfreq = "Measurement Frequencies [GHz]\n{freqs} ".format(freqs=meas_freqs)
        return ("<{name}({insts}{qfreq}\n{mfreq})>"
                "".format(name=self.__class__.__name__, insts=str(self.instruction_schedules),
                          qfreq=qfreq, mfreq=mfreq))

    def build_cmd_def(self) -> 'InstructionScheduleMap':
        """
        Return the InstructionScheduleMap built for this PulseDefaults instance.

        Returns:
            InstructionScheduleMap: `InstructionScheduleMap` instance generated from defaults
        """
        warnings.warn("This method is deprecated. Returning a InstructionScheduleMap instead. "
                      "This can be accessed simply through the `instruction_schedules` attribute "
                      "of this PulseDefaults instance.", DeprecationWarning)
        return self.instruction_schedules


class InstructionScheduleMap():
    """Mapping from QuantumCircuit Instructions to Schedules.
    """

    def __init__(self):
        """Initialize an mapper instance, with this mapping scheme:

            Dict[str, Dict[Tuple[int], Schedule]]

        where the first key is the name of a circuit instruction (e.g. 'u1', 'measure'), the
        second key is a tuple of qubit indices, and the final value is a Schedule implementing
        the requested instruction.
        """
        # The processed and reformatted circuit instruction definitions
        self._map = defaultdict(dict)
        # A backwards mapping from qubit to supported instructions
        self._qubit_insts = defaultdict(set)

    @property
    def instructions(self) -> List[str]:
        """
        Return all instructions which have definitions. By default, these are typically the basis
        gates along with other instructions such as measure and reset.

        Returns:
            The names of all the circuit instructions which have Schedule definitions in this.
        """
        return list(self._map.keys())

    def qubits_with_inst(self, instruction: str) -> List[Union[int, Tuple[int]]]:
        """
        Return a list of the qubits for which the given instruction is defined. Single qubit
        instructions return a flat list, and multiqubit instructions return a list of ordered
        tuples.

        Args:
            instruction: The name of the circuit instruction.

        Returns:
            Qubit indices which have the given instruction defined. This is a list of tuples if the
            instruction has an arity greater than 1, or a flat list of ints otherwise.
        Raises:
            PulseError: If the instruction is not found.
        """
        if instruction not in self._map:
            raise PulseError("No instruction named: {inst}".format(inst=instruction))
        return [qubits[0] if len(qubits) == 1 else qubits
                for qubits in sorted(self._map[instruction].keys())]

    def qubit_insts(self, qubits: Union[int, Iterable[int]]) -> Set[str]:
        """
        Return a list of the instruction names that are defined by the backend for the given qubit
        or qubits.

        Args:
            qubits: A qubit index, or a list or tuple of indices.

        Returns:
            All the instructions which are defined on the qubits. For 1 qubit, all the 1Q
            instructions defined. For multiple qubits, all the instructions which apply to that
            whole set of qubits (e.g. qubits=[0, 1] may return ['cx']).
        Raises:
            PulseError: If the given qubits are not found.
        """
        if _to_tuple(qubits) not in self._qubit_insts:
            raise PulseError("No instructions on qubits: {qubit}".format(qubit=qubits))
        return self._qubit_insts[_to_tuple(qubits)]

    def has(self, instruction: str, qubits: Union[int, Iterable[int]]) -> bool:
        """
        Is the instruction defined for the given qubits?

        Args:
            instruction: The instruction for which to look.
            qubits: The specific qubits for the instruction.

        Returns:
            True iff the instruction is defined.
        """
        return instruction in self._map and \
            _to_tuple(qubits) in self._map[instruction]

    def assert_has(self, instruction: str, qubits: Union[int, Iterable[int]]) -> None:
        """
        Convenience method to check that the given instruction is defined, and error if it is not.

        Args:
            instruction: The instruction for which to look.
            qubits: The specific qubits for the instruction.

        Returns:
            None

        Raises:
            PulseError: If the instruction is not defined on the qubits.
        """
        if not self.has(instruction, _to_tuple(qubits)):
            if instruction in self._map:
                raise PulseError("Operation '{inst}' exists, but is only defined for qubits "
                                 "{qubits}.".format(inst=instruction,
                                                    qubits=self.qubits_with_inst(instruction)))
            raise PulseError("Operation '{inst}' is not defined for this "
                             "system.".format(inst=instruction))

    def get(self,
            instruction: str,
            qubits: Union[int, Iterable[int]],
            *params: List[Union[int, float, complex]],
            **kwparams: Dict[str, Union[int, float, complex]]) -> Schedule:
        """
        Return the defined Schedule for the given instruction on the given qubits.

        Args:
            instruction: Name of the instruction.
            qubits: The qubits for the instruction.
            *params: Command parameters for generating the output schedule.
            **kwparams: Keyworded command parameters for generating the schedule.

        Returns:
            The Schedule defined for the input.
        """
        self.assert_has(instruction, qubits)
        schedule = self._map[instruction].get(_to_tuple(qubits))
        if isinstance(schedule, ParameterizedSchedule):
            schedule = schedule.bind_parameters(*params, **kwparams)
        return schedule

    def get_parameters(self, instruction: str, qubits: Union[int, Iterable[int]]) -> Tuple[str]:
        """
        Return the list of parameters taken by the given instruction on the given qubits.

        Args:
            instruction: Name of the instruction.
            qubits: The qubits for the instruction.

        Returns:
            The names of the parameters required by the instruction.
        """
        self.assert_has(instruction, qubits)
        return self._map[instruction][_to_tuple(qubits)].parameters

    def add(self,
            instruction: str,
            qubits: Union[int, Iterable[int]],
            schedule: [Schedule, ParameterizedSchedule]) -> None:
        """
        Add a new known instruction.

        Args:
            instruction: The name of the instruction to add.
            qubits: The qubits which the instruction applies to.
            schedule: The Schedule that implements the given instruction.

        Returns:
            None

        Raises:
            PulseError: If the qubits are provided as an empty iterable.
        """
        qubits = _to_tuple(qubits)
        if qubits == ():
            raise PulseError("Cannot add definition {} with no target qubits.".format(instruction))
        if not isinstance(schedule, (Schedule, ParameterizedSchedule)):
            raise PulseError("Attemping to add an invalid schedule type.")
        self._map[instruction][qubits] = schedule
        self._qubit_insts[qubits].add(instruction)

    def remove(self, instruction: str, qubits: Union[int, Iterable[int]]) -> None:
        """Remove the given instruction from the defined instructions.

        Args:
            instruction: The name of the instruction to add.
            qubits: The qubits which the instruction applies to.

        Returns:
            None
        """
        qubits = _to_tuple(qubits)
        self.assert_has(instruction, qubits)
        self._map[instruction].pop(qubits)
        self._qubit_insts[qubits].remove(instruction)
        if not self._map[instruction]:
            self._map.pop(instruction)
        if not self._qubit_insts[qubits]:
            self._qubit_insts.pop(qubits)

    def pop(self,
            instruction: str,
            qubits: Union[int, Iterable[int]],
            *params: List[Union[int, float, complex]],
            **kwparams: Dict[str, Union[int, float, complex]]) -> Schedule:
        """
        Remove and return the defined Schedule for the given instruction on the given qubits.

        Args:
            instruction: Name of the instruction.
            qubits: The qubits for the instruction.
            *params: Command parameters for generating the output schedule.
            **kwparams: Keyworded command parameters for generating the schedule.

        Returns:
            The Schedule defined for the input.
        """
        self.assert_has(instruction, qubits)
        schedule = self._map[instruction][_to_tuple(qubits)]
        if isinstance(schedule, ParameterizedSchedule):
            return schedule.bind_parameters(*params, **kwparams)
        self.remove(instruction, qubits)
        return schedule

    def cmds(self) -> List[str]:
        """
        Deprecated.

        Returns:
            The names of all the circuit instructions which have Schedule definitions in this.
        """
        warnings.warn("Please use the `instructions` attribute instead of `cmds()`.",
                      DeprecationWarning)
        return self.instructions

    def cmd_qubits(self, cmd_name: str) -> List[Union[int, Tuple[int]]]:
        """
        Deprecated.

        Args:
            cmd_name: The name of the circuit instruction.

        Returns:
            Qubit indices which have the given instruction defined. This is a list of tuples if
            the instruction has an arity greater than 1, or a flat list of ints otherwise.
        """
        warnings.warn("Please use qubits_with_inst() instead of cmd_qubits().", DeprecationWarning)
        return self.qubits_with_inst(cmd_name)

    def __str__(self):
        single_q_insts = "1Q instructions:\n"
        multi_q_insts = "Multi qubit instructions:\n"
        for qubits, insts in self._qubit_insts.items():
            if len(qubits) == 1:
                single_q_insts += "  q{qubit}: {insts}\n".format(qubit=qubits[0], insts=insts)
            else:
                multi_q_insts += "  {qubits}: {insts}\n".format(qubits=qubits, insts=insts)
        instructions = single_q_insts + multi_q_insts
        return ("<{name}({insts})>"
                "".format(name=self.__class__.__name__, insts=instructions))


def _to_tuple(values):
    """
    Return the input as a tuple, even if it is an integer.

    Args:
        values (Union[int, Iterable[int]]): An integer, or iterable of integers.
    Returns:
        tuple: The input values as a sorted tuple.
    """
    try:
        return tuple(values)
    except TypeError:
        return (values,)
