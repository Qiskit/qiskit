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
import warnings
from collections import defaultdict
from copy import deepcopy
from itertools import zip_longest
from typing import Callable, Iterable, List, Tuple, Union, Optional, NamedTuple

from qiskit.circuit import ParameterExpression
from qiskit.circuit.instruction import Instruction
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ParameterizedSchedule

ScheduleArgumentsTuple = NamedTuple('ScheduleArgumentsTuple',
                                    [('schedule', Union[Callable, Schedule]),
                                     ('arguments', Tuple[str])])
ScheduleArgumentsTuple.__doc__ = 'Set of schedule generator and associated argument names.'
ScheduleArgumentsTuple.schedule.__doc__ = 'Schedule generator function or Schedule.'
ScheduleArgumentsTuple.arguments.__doc__ = 'Name of parameters to be assigned.'


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
        self._map = defaultdict(lambda: defaultdict(ScheduleArgumentsTuple))
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

    def qubits_with_instruction(self,
                                instruction: Union[str, Instruction]) -> List[Union[int,
                                                                                    Tuple[int]]]:
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
        instruction = _get_instruction_string(instruction)
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

    def has(self, instruction: Union[str, Instruction], qubits: Union[int, Iterable[int]]) -> bool:
        """Is the instruction defined for the given qubits?

        Args:
            instruction: The instruction for which to look.
            qubits: The specific qubits for the instruction.

        Returns:
            True iff the instruction is defined.
        """
        instruction = _get_instruction_string(instruction)
        return instruction in self._map and \
            _to_tuple(qubits) in self._map[instruction]

    def assert_has(self,
                   instruction: Union[str, Instruction],
                   qubits: Union[int, Iterable[int]]) -> None:
        """Error if the given instruction is not defined.

        Args:
            instruction: The instruction for which to look.
            qubits: The specific qubits for the instruction.

        Raises:
            PulseError: If the instruction is not defined on the qubits.
        """
        instruction = _get_instruction_string(instruction)
        if not self.has(instruction, _to_tuple(qubits)):
            if instruction in self._map:
                raise PulseError("Operation '{inst}' exists, but is only defined for qubits "
                                 "{qubits}.".format(
                                     inst=instruction,
                                     qubits=self.qubits_with_instruction(instruction)))
            raise PulseError("Operation '{inst}' is not defined for this "
                             "system.".format(inst=instruction))

    def get(self,
            instruction: Union[str, Instruction],
            qubits: Union[int, Iterable[int]],
            *params: Union[int, float, complex, ParameterExpression],
            **kwparams: Union[int, float, complex, ParameterExpression]) -> Schedule:
        """Return the defined :py:class:`~qiskit.pulse.Schedule` for the given instruction on
        the given qubits.

        If all keys are not specified this method returns schedule with unbound parameters.

        Args:
            instruction: Name of the instruction or the instruction itself.
            qubits: The qubits for the instruction.
            *params: Command parameters for generating the output schedule.
            **kwparams: Keyworded command parameters for generating the schedule.

        Returns:
            The Schedule defined for the input.

        Raises:
            PulseError: When invalid parameters are specified.
        """
        instruction = _get_instruction_string(instruction)
        self.assert_has(instruction, qubits)
        schedule_args_tuple = self._map[instruction][_to_tuple(qubits)]

        # Verify parameter-value mapping
        if len(params) > len(schedule_args_tuple.arguments):
            raise PulseError('Too many values to bind: {}.'.format(', '.join(map(str, params))))
        if not all(key in schedule_args_tuple.arguments for key in kwparams):
            raise PulseError('Parameters not defined: {}'.format(', '.join(kwparams.keys())))

        bind_parameters = dict(zip_longest(schedule_args_tuple.arguments, params))
        bind_parameters.update(kwparams)

        sched = schedule_args_tuple.schedule

        # callback function
        if callable(sched):
            return sched(**bind_parameters)

        # schedule
        if sched.is_parameterized():
            parameter_mapping = dict()
            for param_obj in sched.parameters:
                bind_value = bind_parameters[param_obj.name]
                # if value is not set, keep the parameter unassigned
                if bind_value is not None:
                    parameter_mapping[param_obj] = bind_value
            return deepcopy(sched).assign_parameters(parameter_mapping)

        return sched

    def add(self,
            instruction: Union[str, Instruction],
            qubits: Union[int, Iterable[int]],
            schedule: Union[Schedule, Callable[..., Schedule]],
            arguments: Optional[List[str]] = None) -> None:
        """Add a new known instruction for the given qubits and its mapping to a pulse schedule.

        Args:
            instruction: The name of the instruction to add.
            qubits: The qubits which the instruction applies to.
            schedule: The Schedule that implements the given instruction.
            arguments: List of parameter names to create a parameter-bound schedule from the
                associated gate instruction. If :py:meth:`get` is called with arguments rather
                than keyword arguments, this parameter list is used to map the input arguments to
                parameter objects stored in the target schedule.

        Raises:
            PulseError: If the qubits are provided as an empty iterable.
        """
        instruction = _get_instruction_string(instruction)

        # validation of target qubit
        qubits = _to_tuple(qubits)
        if qubits == ():
            raise PulseError("Cannot add definition {} with no target qubits.".format(instruction))

        # TODO this block will be removed
        if isinstance(schedule, ParameterizedSchedule):
            warnings.warn('ParameterizedSchedule has been deprecated. '
                          'Define Schedule with Parameter objects.', DeprecationWarning)

            def sched_callback(**kwargs):
                bind_dict = {pname: kwargs[pname] for pname in schedule.parameters}
                return schedule.bind_parameters(**bind_dict)
            arguments = tuple(schedule.parameters)

            self._map[instruction][qubits] = ScheduleArgumentsTuple(sched_callback, arguments)
            self._qubit_instructions[qubits].add(instruction)

            return

        # validation of input data
        if not (isinstance(schedule, Schedule) or callable(schedule)):
            raise PulseError('Supplied schedule must be either a Schedule, or a '
                             'callable that outputs a schedule.')

        # initialize parameter list
        if callable(schedule):
            func_parameters = list(inspect.signature(schedule).parameters.keys())
        else:
            func_parameters = set(param.name for param in schedule.parameters)

        if arguments is None:
            # for backward compatibility
            arguments = sorted(func_parameters)
        else:
            # check parameter list consistency
            if sorted(func_parameters) != sorted(arguments):
                str_func_parameters = ', '.join(func_parameters)
                str_arguments = ', '.join(arguments)
                raise PulseError('Program signature and specified parameter names do not match '
                                 '{} != {}'.format(str_func_parameters, str_arguments))

        self._map[instruction][qubits] = ScheduleArgumentsTuple(schedule, tuple(arguments))
        self._qubit_instructions[qubits].add(instruction)

    def remove(self,
               instruction: Union[str, Instruction],
               qubits: Union[int, Iterable[int]]) -> None:
        """Remove the given instruction from the listing of instructions defined in self.

        Args:
            instruction: The name of the instruction to add.
            qubits: The qubits which the instruction applies to.
        """
        instruction = _get_instruction_string(instruction)
        qubits = _to_tuple(qubits)
        self.assert_has(instruction, qubits)
        self._map[instruction].pop(qubits)
        self._qubit_instructions[qubits].remove(instruction)
        if not self._map[instruction]:
            self._map.pop(instruction)
        if not self._qubit_instructions[qubits]:
            self._qubit_instructions.pop(qubits)

    def pop(self,
            instruction: Union[str, Instruction],
            qubits: Union[int, Iterable[int]],
            *params: Union[int, float, complex, ParameterExpression],
            **kwparams: Union[int, float, complex, ParameterExpression]) -> Schedule:
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
        instruction = _get_instruction_string(instruction)
        schedule = self.get(instruction, qubits, *params, **kwparams)
        self.remove(instruction, qubits)
        return schedule

    def get_parameters(self,
                       instruction: Union[str, Instruction],
                       qubits: Union[int, Iterable[int]]
                       ) -> Tuple[Union[str, ParameterExpression]]:
        """Return the list of parameters taken by the given instruction on the given qubits.

        Args:
            instruction: Name of the instruction.
            qubits: The qubits for the instruction.

        Returns:
            The names of the parameters required by the instruction.
        """
        instruction = _get_instruction_string(instruction)

        self.assert_has(instruction, qubits)
        return self._map[instruction][_to_tuple(qubits)].arguments

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


def _get_instruction_string(inst: Union[str, Instruction]):
    if isinstance(inst, str):
        return inst
    else:
        try:
            return inst.name
        except AttributeError as ex:
            raise PulseError('Input "inst" has no attribute "name".'
                             'This should be a circuit "Instruction".') from ex
