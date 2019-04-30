# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Command definition module. Relates circuit gates to pulse commands.
"""
from typing import List, Tuple, Iterable, Union, Dict

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.qobj import PulseLibraryItem

from .commands import (SamplePulse, PersistentValue, Acquire, FrameChange,
                       PulseInstruction, FrameChangeInstruction, AcquireInstruction,
                       PersistentValueInstruction)

from .exceptions import PulseError
from .schedule import Schedule


def build_pulse_library(self, pulse_library: List[PulseLibraryItem]):
    """Take pulse library and convert to dictionary of `SamplePulse`s.

    Args:
        pulse_library: Unprocessed pulse_library.

    Returns:
        dict: Pulse library consisting of `SamplePulse`s
    """
    processed_pulse_library = {}

    for pulse in pulse_library:
        if isinstance(pulse, dict):
            pulse = PulseLibraryItem(**pulse)

        name = pulse.name
        pulse = SamplePulse(np.asarray(pulse.samples, dtype=np.complex_), name=name)
        processed_pulse_library[name] = pulse

    return processed_pulse_library


def _to_qubit_tuple(qubit_tuple: Union[int, Iterable[int]]):
    """Convert argument to tuple.
    Args:
        qubit_tuple: Qubits to enforce as tuple.

    Returns:
        tuple
    """
    try:
        qubit_tuple = tuple(qubit_tuple)
    except TypeError:
        qubit_tuple = (qubit_tuple,)

    if not all(isinstance(i, int) for i in qubit_tuple):
        raise QiskitError("All qubits must be integers.")


class ParameterizedSchedule:
    """Temporary parameterized schedule class.

    This should not be returned to users as it is currently only a helper class.

    This class is takes an input command definition that accepts
    a set of parameters. Calling `bind` on the class will return a `Schedule`.

    # TODO: In the near future this will be replaced with proper incorporation of parameters
        into the `Schedule` class.
    """

    def __init__(commands):
        pass

    @property
    def paramaters(self) -> Tuple[str]:
        """Schedule parameters."""
        pass

    def bind_parameters(**params: Iterable[str, Union[float, complex]]) -> Schedule:
        """Generate the Schedule from params to evaluate command expressions

        Args:
            *params:
        """
        pass


class CmdDef:
    """Command definition class.
    Relates `Gate`s to `PulseSchedule`s.
    """

    def __init__(self, schedules=None):
        """Create command definition from backend.

        Args:
            dict: Keys are tuples of (cmd_name, *qubits) and values are `PulseSchedule`
        """
        self._cmd_dict = {}

        if schedules:
            for key, schedule in schedules.items():
                self.add(key[0], key[1:], schedule)

    @classmethod
    def from_defaults(cls, flat_cmd_def: List[PulseQobjInstruction],
                      pulse_library: Dict[str, SamplePulse]) -> 'CmdDef':
        """Create command definition from backend defaults output.
        Args:
            flat_cmd_def: Command definition list returned by backend
            pulse_library: Dictionary of SamplePulses.
        """
        pulse_library = build_pulse_library(pulse_library)

    def add(self, cmd_name: str, qubits: Union[int, Iterable[int]],
            schedule: Union[ParameterizedSchedule, Schedule]):
        """Add a command to the `CommandDefinition`

        Args:
            cmd_name: Name of the command
            qubits: Qubits command applies to
            schedule: Schedule to be added
        """
        qubits = _to_qubit_tuple(qubits)
        cmd_dict = self._cmd_dict.setdefault(schedule, {})
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
            **params: Iterable[str, Union[float, complex]]) -> Schedule:
        """Get command from command definition.
        Args:
            cmd_name: Name of the command
            qubits: Ordered list of qubits command applies to
            **params: Command parameters to be used to generate schedule

        Raises:
            PulseError
        """
        qubits = _to_qubit_tuple(qubits)
        if self.has_cmd(cmd_name, qubits):
            return self._cmd_dict[cmd_name][qubits]
        else:
            raise PulseError('Command {name} for qubits {qubits} is not present'
                             'in CmdDef'.format(cmd_name, qubits))

    def pop(self, cmd_name: str, qubits: Union[int, Iterable[int]],
            **params: Iterable[str, Union[float, complex]]) -> Schedule:
        """Pop command from command definition.

        Args:
            cmd_name (str): Name of the command
            qubits (int, list or tuple): Ordered list of qubits command applies to
            **params: Command parameters to be used to generate schedule
        """
        qubits = _to_qubit_tuple(qubits)
        if self.has_cmd(cmd_name, qubits):
            cmd_dict = self._cmd_dict[cmd_name]
            cmd = cmd_dict.pop(qubits)
            if not cmd_dict:
                self._cmd_dict.pop(cmd_name)
            return cmd.bind(**params)
        else:
            raise PulseError('Command {name} for qubits {qubits} is not present'
                             'in CmdDef'.format(cmd_name, qubits))

    def cmd_types(self) -> List[str]:
        """Return all command names available in CmdDef."""

        return list(self._cmd_dict.keys())

    def cmd_qubits(self, cmd_name: str) -> List[Tuple[int]]:
        """Get all qubit orderings this command exists for."""
        if cmd_name in self._cmd_dict:
            return list(self._cmd_dict.keys())
        raise PulseError('Command %s does not exist in CmdDef.' % cmd_name)

    def __repr__(self):
        return repr(self._cmd_dict)
