# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Command definition module. Relates circuit gates to pulse commands.
"""
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.qobj import PulseLibraryItem

from .commands import (SamplePulse, PersistentValue, Acquire, FrameChange,
                       PulseInstruction, FrameChangeInstruction, AcquireInstruction,
                       PersistentValueInstruction)

from .exceptions import PulseError
from .schedule import Schedule


def _preprocess_pulse_library(self, pulse_library: PulseLibraryItem):
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


def _to_qubit_tuple(qubit_tuple):
    """Convert argument to tuple.
    Args:
        qubit_tuple (int, iterable): qubits to enforce as tuple.

    Returns:
        tuple
    """
    try:
        qubit_tuple = tuple(qubit_tuple)
    except TypeError:
        qubit_tuple = (qubit_tuple,)

    if not all(isinstance(i, int) for i in qubit_tuple):
        raise QiskitError("All qubits must be integers.")


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
    def cmd_def_from_defaults(cls, flat_cmd_def, pulse_library):
        """Create command definition from backend defaults output.
        Args:
            flat_cmd_def (list): Command definition list returned by backend
            pulse_library (list): dict or list of pulse library entries
        Returns:
            CmdDef
        """
        pulse_library = _preprocess_pulse_library(pulse_library)

    def add(self, cmd_name, qubits, schedule):
        """Add a command to the `CommandDefinition`

        Args:
            cmd_name (str): Name of the command
            qubits (int, list or tuple): Ordered list of qubits command applies to
            schedule (ParameterizedSchedule or Schedule): Schedule to be added
        """
        qubits = _to_qubit_tuple(qubits)
        cmd_dict = self._cmd_dict.setdefault('cmd', {})
        cmd_dict[qubits] = schedule

    def has_cmd(self, cmd_name, qubits):
        """Has command of name with qubits.

        Args:
            cmd_name (str): Name of the command
            qubits (int, list or tuple): Ordered list of qubits command applies to

        Returns:
            bool
        """
        qubits = _to_qubit_tuple(qubits)
        if cmd_name in self._cmd_dict:
            if qubits in self._cmd_dict[cmd_name]:
                return True
        return False

    def get(self, cmd_name, qubits, default=None):
        """Get command from command definition.
        Args:
            cmd_name (str): Name of the command
            qubits (int, list or tuple): Ordered list of qubits command applies to
            default (None or ParameterizedPulseSchedule or PulseSchedule): Default PulseSchedule
                to return if command is not in CmdDef.
        Returns:
            PulseSchedule or ParameterizedPulseSchedule

        Raises:
            ScheduleError
        """
        qubits = _to_qubit_tuple(qubits)
        if self.has_cmd(cmd_name, qubits):
            return self._cmd_dict[cmd_name][qubits]
        elif default:
                return default
        else:
            raise PulseError('Command {name} for qubits {qubits} is not present'
                             'in CmdDef'.format(cmd_name, qubits))

    def pop(self, cmd_name, qubits, default=None):
        """Pop command from command definition.

        Args:
            cmd_name (str): Name of the command
            qubits (int, list or tuple): Ordered list of qubits command applies to
            default (None or ParameterizedPulseSchedule or PulseSchedule): Default PulseSchedule
                to return if command is not in CmdDef.
        Returns:
            PulseSchedule or ParameterizedPulseSchedule

        Raises:
            ScheduleError
        """
        qubits = _to_qubit_tuple(qubits)
        if self.has_cmd(cmd_name, qubits):
            cmd_dict = self._cmd_dict[cmd_name]
            cmd = cmd_dict.pop(qubits)
            if not cmd_dict:
                self._cmd_dict.pop(cmd_name)
            return cmd
        elif default:
            return default
        else:
            raise PulseError('Command {name} for qubits {qubits} is not present'
                             'in CmdDef'.format(cmd_name, qubits))

    def cmd_types(self):
        """Return all command names available in CmdDef.

        Returns:
            list
        """

        return list(self._cmd_dict.keys())

    def cmds(self):
        """Returns list of all commands.

        Returns:
            (str, tuple, PulseSchedule or ParameterizedPulseSchedule): A tuple containing
                the command name, tuple of qubits the command applies to and the command
                schedule.
        """
        return list(self)

    def __repr__(self):
        return repr(self._cmd_dict)
