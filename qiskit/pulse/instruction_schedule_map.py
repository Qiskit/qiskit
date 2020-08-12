# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
A convenient way to track reusable subschedules by name and qubit.

This can be used for scheduling circuits with custom definitions, for instance::

    inst_map = InstructionScheduleMap()
    inst_map.add('new_inst', 0, qubit_0_new_inst_schedule)

    sched = schedule(quantum_circuit, backend, inst_map)

An instance of this class is instantiated by Pulse-enabled backends and populated with defaults
(if available)::

    inst_map = backend.defaults().instruction_schedule_map

"""
import inspect

from collections import defaultdict
from typing import List, Tuple, Iterable, Union, Callable

from .schedule import Schedule, ParameterizedSchedule
from .exceptions import PulseError


class InstructionScheduleMap():
    """Mapping from :py:class:`~qiskit.circuit.QuantumCircuit`
    :py:class:`qiskit.circuit.Instruction` names and qubits to
    :py:class:`~qiskit.pulse.Schedule` s. In particular, the mapping is formatted as type::

         Dict[str, Dict[Tuple[int], Schedule]]

    where the first key is the name of a circuit instruction (e.g. ``'u1'``, ``'measure'``), the
    second key is a tuple of qubit indices, and the final value is a Schedule implementing the
    requested instruction.

    These can usually be seen as gate calibrations.
    """

    def __init__(self):
        """Initialize a circuit instruction to schedule mapper instance."""
        # The processed and reformatted circuit instruction definitions
        self._map = defaultdict(dict)
        # A backwards mapping from qubit to supported instructions
        self._qubit_instructions = defaultdict(set)

    @property
    def instructions(self) -> List[str]:
        """Return all instructions which have definitions.

        By default, these are typically the basis gates along with other instructions such as
        measure and reset.

        Returns:
            The names of all the circuit instructions which have Schedule definitions in this.
        """
        return list(self._map.keys())

    def qubits_with_instruction(self, instruction: str) -> List[Union[int, Tuple[int]]]:
        """Return a list of the qubits for which the given instruction is defined. Single qubit
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
            return []
        return [qubits[0] if len(qubits) == 1 else qubits
                for qubits in sorted(self._map[instruction].keys())]

    def qubit_instructions(self, qubits: Union[int, Iterable[int]]) -> List[str]:
        """Return a list of the instruction names that are defined by the backend for the given
        qubit or qubits.

        Args:
            qubits: A qubit index, or a list or tuple of indices.

        Returns:
            All the instructions which are defined on the qubits.

            For 1 qubit, all the 1Q instructions defined. For multiple qubits, all the instructions
            which apply to that whole set of qubits (e.g. ``qubits=[0, 1]`` may return ``['cx']``).
        """
        if _to_tuple(qubits) in self._qubit_instructions:
            return list(self._qubit_instructions[_to_tuple(qubits)])
        return []

    def has(self, instruction: str, qubits: Union[int, Iterable[int]]) -> bool:
        """Is the instruction defined for the given qubits?

        Args:
            instruction: The instruction for which to look.
            qubits: The specific qubits for the instruction.

        Returns:
            True iff the instruction is defined.
        """
        return instruction in self._map and \
            _to_tuple(qubits) in self._map[instruction]

    def assert_has(self, instruction: str, qubits: Union[int, Iterable[int]]) -> None:
        """Error if the given instruction is not defined.

        Args:
            instruction: The instruction for which to look.
            qubits: The specific qubits for the instruction.

        Raises:
            PulseError: If the instruction is not defined on the qubits.
        """
        if not self.has(instruction, _to_tuple(qubits)):
            if instruction in self._map:
                raise PulseError("Operation '{inst}' exists, but is only defined for qubits "
                                 "{qubits}.".format(
                                     inst=instruction,
                                     qubits=self.qubits_with_instruction(instruction)))
            raise PulseError("Operation '{inst}' is not defined for this "
                             "system.".format(inst=instruction))

    def get(self,
            instruction: str,
            qubits: Union[int, Iterable[int]],
            *params: Union[int, float, complex],
            **kwparams: Union[int, float, complex]) -> Schedule:
        """Return the defined :py:class:`~qiskit.pulse.Schedule` for the given instruction on
        the given qubits.

        Args:
            instruction: Name of the instruction.
            qubits: The qubits for the instruction.
            *params: Command parameters for generating the output schedule.
            **kwparams: Keyworded command parameters for generating the schedule.

        Returns:
            The Schedule defined for the input.
        """
        self.assert_has(instruction, qubits)
        schedule_generator = self._map[instruction].get(_to_tuple(qubits))

        if callable(schedule_generator):
            return schedule_generator(*params, **kwparams)
        # otherwise this is just a Schedule
        return schedule_generator

    def add(self,
            instruction: str,
            qubits: Union[int, Iterable[int]],
            schedule: Union[Schedule, Callable[..., Schedule]]) -> None:
        """Add a new known instruction for the given qubits and its mapping to a pulse schedule.

        Args:
            instruction: The name of the instruction to add.
            qubits: The qubits which the instruction applies to.
            schedule: The Schedule that implements the given instruction.

        Raises:
            PulseError: If the qubits are provided as an empty iterable.
        """
        qubits = _to_tuple(qubits)
        if qubits == ():
            raise PulseError("Cannot add definition {} with no target qubits.".format(instruction))
        if not (isinstance(schedule, Schedule) or callable(schedule)):
            raise PulseError('Supplied schedule must be either a Schedule, or a '
                             'callable that outputs a schedule.')
        self._map[instruction][qubits] = schedule
        self._qubit_instructions[qubits].add(instruction)

    def remove(self, instruction: str, qubits: Union[int, Iterable[int]]) -> None:
        """Remove the given instruction from the listing of instructions defined in self.

        Args:
            instruction: The name of the instruction to add.
            qubits: The qubits which the instruction applies to.
        """
        qubits = _to_tuple(qubits)
        self.assert_has(instruction, qubits)
        self._map[instruction].pop(qubits)
        self._qubit_instructions[qubits].remove(instruction)
        if not self._map[instruction]:
            self._map.pop(instruction)
        if not self._qubit_instructions[qubits]:
            self._qubit_instructions.pop(qubits)

    def pop(self,
            instruction: str,
            qubits: Union[int, Iterable[int]],
            *params: Union[int, float, complex],
            **kwparams: Union[int, float, complex]) -> Schedule:
        """Remove and return the defined ``Schedule`` for the given instruction on the given
        qubits.

        Args:
            instruction: Name of the instruction.
            qubits: The qubits for the instruction.
            *params: Command parameters for generating the output schedule.
            **kwparams: Keyworded command parameters for generating the schedule.

        Returns:
            The Schedule defined for the input.
        """
        schedule = self.get(instruction, qubits, *params, **kwparams)
        self.remove(instruction, qubits)
        return schedule

    def get_parameters(self, instruction: str, qubits: Union[int, Iterable[int]]) -> Tuple[str]:
        """Return the list of parameters taken by the given instruction on the given qubits.

        Args:
            instruction: Name of the instruction.
            qubits: The qubits for the instruction.

        Returns:
            The names of the parameters required by the instruction.
        """
        self.assert_has(instruction, qubits)
        schedule_generator = self._map[instruction][_to_tuple(qubits)]
        if isinstance(schedule_generator, ParameterizedSchedule):
            return schedule_generator.parameters
        elif callable(schedule_generator):
            return tuple(inspect.signature(schedule_generator).parameters.keys())
        else:
            return ()

    def __str__(self):
        single_q_insts = "1Q instructions:\n"
        multi_q_insts = "Multi qubit instructions:\n"
        for qubits, insts in self._qubit_instructions.items():
            if len(qubits) == 1:
                single_q_insts += "  q{qubit}: {insts}\n".format(qubit=qubits[0], insts=insts)
            else:
                multi_q_insts += "  {qubits}: {insts}\n".format(qubits=qubits, insts=insts)
        instructions = single_q_insts + multi_q_insts
        return ("<{name}({insts})>"
                "".format(name=self.__class__.__name__, insts=instructions))


def _to_tuple(values: Union[int, Iterable[int]]) -> Tuple[int, ...]:
    """Return the input as a tuple.

    Args:
        values: An integer, or iterable of integers.

    Returns:
        The input values as a sorted tuple.
    """
    try:
        return tuple(values)
    except TypeError:
        return (values,)
