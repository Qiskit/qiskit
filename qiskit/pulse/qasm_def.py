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
QASM operation (circuit gates and measures) to Schedule definition module.
"""
from typing import List, Tuple, Iterable, Union, Dict, Optional

import warnings

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


class QasmToSchedDef:
    """Definitions of supported QASM Gates and Measures in terms of Schedules."""

    def __init__(self, schedules: Optional[Dict] = None):
        """Create definitions for gates and measures from backend.

        Args:
            schedules: Keys are tuples of (name, *qubits) and values are
                `Schedule` or `ParameterizedSchedule`
        """
        self._gate_dict = {}

        if schedules:
            for key, schedule in schedules.items():
                self.add(key[0], key[1:], schedule)

    @classmethod
    def from_defaults(cls,
                      flat_qasm_def: List[PulseQobjInstruction],
                      pulse_library: Dict[str, SamplePulse]) -> 'QasmToSchedDef':
        """Create definitions from backend defaults output.

        Args:
            flat_qasm_def: `Command` definition list returned by backend
            pulse_library: Dictionary of `SamplePulse`s
        """
        converter = QobjToInstructionConverter(pulse_library, buffer=0)
        qasm_def = cls()

        for gate in flat_qasm_def:
            qubits = gate.qubits
            name = gate.name
            instructions = []
            for instr in gate.sequence:
                instructions.append(converter(instr))

            qasm_def.add(name, qubits, ParameterizedSchedule(*instructions, name=name))

        return qasm_def

    def add(self, name: str, qubits: Union[int, Iterable[int]],
            schedule: Union[ParameterizedSchedule, Schedule]):
        """Add an operation to this `QasmToSchedDef`

        Args:
            name: Name of the gate or measure
            qubits: Qubits involved in the QASM operation
            schedule: Schedule to be added
        """
        qubits = _to_qubit_tuple(qubits)
        gate_dict = self._gate_dict.setdefault(name, {})
        if isinstance(schedule, Schedule):
            schedule = ParameterizedSchedule(schedule, name=schedule.name)
        gate_dict[qubits] = schedule

    def has(self, name: str, qubits: Union[int, Iterable[int]]) -> bool:
        """Has operation of name with qubits.

        Args:
            name: Name of the gate or measure
            qubits: Qubits involved in the QASM operation
        """
        qubits = _to_qubit_tuple(qubits)
        return name in self._gate_dict and qubits in self._gate_dict[name]

    def get(self,
            name: str,
            qubits: Union[int, Iterable[int]],
            *params: List[Union[int, float, complex]],
            **kwparams: Dict[str, Union[int, float, complex]]) -> Schedule:
        """Get Schedule from QASM operation.

        Args:
            name: Name of the gate or measure
            qubits: Ordered list of qubits to which the gate is applied
            *params: Parameters to be used to generate schedule
            **kwparams: Keyworded parameters to be used to generate schedule

        Raises:
            PulseError: If there is no Schedule defined for the given operation on the given qubits
        """
        qubits = _to_qubit_tuple(qubits)
        if self.has(name, qubits):
            schedule = self._gate_dict[name][qubits]

            if isinstance(schedule, ParameterizedSchedule):
                return schedule.bind_parameters(*params, **kwparams)

            return schedule.flatten()

        else:
            raise PulseError('Command {0} for qubits {1} is not present '
                             'in QasmToSchedDef'.format(name, qubits))

    def get_parameters(self, name: str, qubits: Union[int, Iterable[int]]) -> Tuple[str]:
        """Return the parameters associated with the Schedule defined for the given gate.
        Args:
            name: Name of the gate or measure
            qubits: Qubits involved in the QASM operation

        Raises:
            PulseError: If there is no Schedule defined for the given operation on the given qubits
        """
        qubits = _to_qubit_tuple(qubits)
        if self.has(name, qubits):
            schedule = self._gate_dict[name][qubits]
            return schedule.parameters

        else:
            raise PulseError('Command {0} for qubits {1} is not present '
                             'in QasmToSchedDef'.format(name, qubits))

    def pop(self,
            name: str,
            qubits: Union[int, Iterable[int]],
            *params: List[Union[int, float, complex]],
            **kwparams: Dict[str, Union[int, float, complex]]) -> Schedule:
        """Pop the given QASM operation from this set of definitions.

        Args:
            name: Name of the gate or measure
            qubits: Qubits involved in the QASM operation
            *params: Parameters used to generate the Schedule
            **kwparams: Keyworded parameters used to generate the Schedule

        Raises:
            PulseError: If there is no Schedule defined for the given operation on the given qubits
        """
        qubits = _to_qubit_tuple(qubits)
        if self.has(name, qubits):
            gate_dict = self._gate_dict[name]
            schedule = gate_dict.pop(qubits)

            if isinstance(schedule, ParameterizedSchedule):
                return schedule.bind_parameters(*params, **kwparams)

            return schedule

        else:
            raise PulseError('Command {0} for qubits {1} is not present '
                             'in QasmToSchedDef'.format(name, qubits))

    def gates(self) -> List[str]:
        """Return all gate or measure names available in this QasmToSchedDef."""
        return list(self._gate_dict.keys())

    def gate_qubits(self, name: str) -> List[Tuple[int]]:
        """Get all qubit groups this gate or measure exists for."""
        if name in self._gate_dict:
            return list(sorted(self._gate_dict[name].keys()))
        return []

    def cmds(self) -> List[str]:
        """Return all command names available in QasmToSchedDef."""
        warnings.warn("Method `cmds` deprecated, use `gates` instead.")
        return list(self._gate_dict.keys())

    def cmd_qubits(self, name: str) -> List[Tuple[int]]:
        """Get all qubit orderings this command exists for."""
        warnings.warn("Method `cmd_qubits` deprecated, use `gate_qubits` instead.")
        if name in self._gate_dict:
            return list(sorted(self._gate_dict[name].keys()))
        return []

    def __repr__(self):
        return repr(self._gate_dict)
