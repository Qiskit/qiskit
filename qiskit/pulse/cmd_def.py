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

"""
Command definition module. Relates circuit gates to pulse commands.
"""
import warnings

from typing import List, Tuple, Iterable, Union, Dict, Optional

from qiskit.qobj import PulseQobjInstruction
from qiskit.qobj.converters import QobjToInstructionConverter

from .commands import SamplePulse

from .exceptions import PulseError
from .schedule import Schedule, ParameterizedSchedule

# pylint: disable=missing-return-doc


def _to_qubit_tuple(qubit_tuple: Union[int, Iterable[int]]) -> Tuple[int]:
    """Convert argument to tuple.

    Args:
        qubit_tuple: Qubits to enforce as tuple.

    Raises:
        PulseError: If qubits are not integers
    """
    try:
        qubit_tuple = tuple(qubit_tuple)
    except TypeError:
        qubit_tuple = (qubit_tuple,)

    if not all(isinstance(i, int) for i in qubit_tuple):
        raise PulseError("All qubits must be integers.")

    return qubit_tuple


class CmdDef:
    """Command definition class. Relates `Gate`s to `Schedule`s."""

    def __init__(self, schedules: Optional[Dict] = None):
        """Create command definition from backend.

        Args:
            schedules: Keys are tuples of (cmd_name, *qubits) and values are
                `Schedule` or `ParameterizedSchedule`
        """
        warnings.warn("The CmdDef is being deprecated. All CmdDef methods are now supported by "
                      "`InstructionScheduleMap` accessible as "
                      "`backend.defaults().instruction_schedule_map` for any Pulse enabled system.",
                      DeprecationWarning)
        self._cmd_dict = {}

        if schedules:
            for key, schedule in schedules.items():
                self.add(key[0], key[1:], schedule)

    @classmethod
    def from_defaults(cls, flat_cmd_def: List[PulseQobjInstruction],
                      pulse_library: Dict[str, SamplePulse],
                      buffer: int = 0) -> 'CmdDef':
        """Create command definition from backend defaults output.

        Args:
            flat_cmd_def: Command definition list returned by backend
            pulse_library: Dictionary of `SamplePulse`s
            buffer: Buffer between instructions on channel
        """
        if buffer:
            warnings.warn("Buffers are no longer supported. Please use an explicit Delay.")
        converter = QobjToInstructionConverter(pulse_library)
        cmd_def = cls()

        for cmd in flat_cmd_def:
            qubits = cmd.qubits
            name = cmd.name
            instructions = []
            for instr in cmd.sequence:
                instructions.append(converter(instr))

            cmd_def.add(name, qubits, ParameterizedSchedule(*instructions, name=name))

        return cmd_def

    def add(self, cmd_name: str, qubits: Union[int, Iterable[int]],
            schedule: Union[ParameterizedSchedule, Schedule]):
        """Add a command to the `CommandDefinition`

        Args:
            cmd_name: Name of the command
            qubits: Qubits command applies to
            schedule: Schedule to be added
        """
        qubits = _to_qubit_tuple(qubits)
        cmd_dict = self._cmd_dict.setdefault(cmd_name, {})
        if isinstance(schedule, Schedule):
            schedule = ParameterizedSchedule(schedule, name=schedule.name)
        cmd_dict[qubits] = schedule

    def has(self, cmd_name: str, qubits: Union[int, Iterable[int]]) -> bool:
        """Has command of name with qubits.

        Args:
            cmd_name: Name of the command
            qubits: Ordered list of qubits command applies to
        """
        qubits = _to_qubit_tuple(qubits)
        if cmd_name in self._cmd_dict:

            if qubits in self._cmd_dict[cmd_name]:
                return True

        return False

    def get(self, cmd_name: str, qubits: Union[int, Iterable[int]],
            *params: List[Union[int, float, complex]],
            **kwparams: Dict[str, Union[int, float, complex]]) -> Schedule:
        """Get command from command definition.

        Args:
            cmd_name: Name of the command
            qubits: Ordered list of qubits command applies to
            *params: Command parameters to be used to generate schedule
            **kwparams: Keyworded command parameters to be used to generate schedule

        Raises:
            PulseError: If command for qubits is not available
        """
        qubits = _to_qubit_tuple(qubits)
        if self.has(cmd_name, qubits):
            schedule = self._cmd_dict[cmd_name][qubits]

            if isinstance(schedule, ParameterizedSchedule):
                return schedule.bind_parameters(*params, **kwparams)

            return schedule.flatten()

        else:
            raise PulseError('Command {0} for qubits {1} is not present '
                             'in CmdDef'.format(cmd_name, qubits))

    def get_parameters(self, cmd_name: str, qubits: Union[int, Iterable[int]]) -> Tuple[str]:
        """Get command parameters from command definition.

        Args:
            cmd_name: Name of the command
            qubits: Ordered list of qubits command applies to

        Raises:
            PulseError: If command for qubits is not available
        """
        qubits = _to_qubit_tuple(qubits)
        if self.has(cmd_name, qubits):
            schedule = self._cmd_dict[cmd_name][qubits]
            return schedule.parameters

        else:
            raise PulseError('Command {0} for qubits {1} is not present '
                             'in CmdDef'.format(cmd_name, qubits))

    def pop(self, cmd_name: str, qubits: Union[int, Iterable[int]],
            *params: List[Union[int, float, complex]],
            **kwparams: Dict[str, Union[int, float, complex]]) -> Schedule:
        """Pop command from command definition.

        Args:
            cmd_name: Name of the command
            qubits: Ordered list of qubits command applies to
            *params: Command parameters to be used to generate schedule
            **kwparams: Keyworded command parameters to be used to generate schedule

        Raises:
            PulseError: If command for qubits is not available
        """
        qubits = _to_qubit_tuple(qubits)
        if self.has(cmd_name, qubits):
            cmd_dict = self._cmd_dict[cmd_name]
            schedule = cmd_dict.pop(qubits)

            if isinstance(schedule, ParameterizedSchedule):
                return schedule.bind_parameters(*params, **kwparams)

            return schedule

        else:
            raise PulseError('Command {0} for qubits {1} is not present '
                             'in CmdDef'.format(cmd_name, qubits))

    def cmds(self) -> List[str]:
        """Return all command names available in CmdDef."""

        return list(self._cmd_dict.keys())

    def cmd_qubits(self, cmd_name: str) -> List[Tuple[int]]:
        """Get all qubit orderings this command exists for."""
        if cmd_name in self._cmd_dict:
            return list(sorted(self._cmd_dict[cmd_name].keys()))

        return []

    def __repr__(self):
        return repr(self._cmd_dict)
