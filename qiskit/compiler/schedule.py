# -*- coding: utf-8 -*-

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
Convenience entry point into pulse scheduling, requiring only a circuit and a backend. For more
control over pulse scheduling, look at `qiskit.scheduler.schedule_circuit`.
"""
from typing import List, Optional, Union

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.pulse.cmd_def import CmdDef
from qiskit.pulse.schedule import Schedule

from qiskit.scheduler import schedule_circuit, ScheduleConfig


def schedule(circuits: Union[QuantumCircuit, List[QuantumCircuit]],
             backend: Optional['BaseBackend'] = None,
             cmd_def: Optional[CmdDef] = None,
             meas_map: Optional[List[List[int]]] = None,
             method: Optional[Union[str, List[str]]] = None) -> Union[Schedule, List[Schedule]]:
    """
    Schedule a circuit to a pulse Schedule, using the backend, according to any specified methods.
    Supported methods are documented in
    :py:func:`qiskit.pulse.scheduler.schedule_circuit.schedule_circuit`.

    Args:
        circuits: The quantum circuit or circuits to translate
        backend: A backend instance, which contains hardware-specific data required for scheduling
        cmd_def: Mapping of circuit operations to pulse schedules. If None, defaults to the
                 `backend` `cmd_def`
        meas_map: List of sets of qubits that must be measured together. If `None`, defaults to
                  the `backend` `meas_map`
        method: Optionally specify a particular scheduling method
    Returns:
        A pulse `Schedule` that implements the input circuit
    Raises:
        QiskitError: If `cmd_def` and `meas_map` are not passed and `backend` is not passed
    """
    if cmd_def is None:
        if backend is None:
            raise QiskitError("Must supply either a backend or CmdDef for scheduling passes.")
        defaults = backend.defaults()
        cmd_def = CmdDef.from_defaults(defaults.cmd_def, defaults.pulse_library)
    if meas_map is None:
        if backend is None:
            raise QiskitError("Must supply either a backend or a meas_map for scheduling passes.")
        meas_map = backend.configuration().meas_map

    schedule_config = ScheduleConfig(cmd_def=cmd_def, meas_map=meas_map)
    circuits = circuits if isinstance(circuits, list) else [circuits]
    schedules = [schedule_circuit(circuit, schedule_config, method) for circuit in circuits]
    return schedules[0] if len(schedules) == 1 else schedules
