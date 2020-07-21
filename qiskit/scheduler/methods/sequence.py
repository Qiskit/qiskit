# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Mapping a scheduled QuantumCircuit to a pulse Schedule.
"""

from qiskit.circuit.barrier import Barrier
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.pulse.schedule import Schedule
from qiskit.pulse.transforms import pad
from qiskit.scheduler.config import ScheduleConfig
from qiskit.scheduler.methods.lowering import lower_gates


def sequence(scheduled_circuit: QuantumCircuit, schedule_config: ScheduleConfig) -> Schedule:
    """
    Return the pulse Schedule which implements the input scheduled circuit.

    Assume all measurements are done at once at the last of the circuit.
    Schedules according to the command definition given by the schedule_config.

    Args:
        scheduled_circuit: The scheduled quantum circuit to translate.
        schedule_config: Backend specific parameters used for building the Schedule.

    Returns:
        A schedule corresponding to the input ``circuit``.

    Raises:
        QiskitError: If invalid scheduled circuit is supplied.
    """
    circ_pulse_defs = lower_gates(scheduled_circuit, schedule_config)

    # restore start times
    qubit_time_available = {}
    start_times = []
    for circ_pulse_def in circ_pulse_defs:
        active_qubits = [q for q in circ_pulse_def.qubits if q in qubit_time_available]

        start_time = max([qubit_time_available[q] for q in active_qubits], default=0)
        start_times.append(start_time)

        for q in active_qubits:
            if qubit_time_available[q] != start_time:
                raise QiskitError("Invalid scheduled circuit.")

        stop_time = start_time
        if not isinstance(circ_pulse_def.schedule, Barrier):
            stop_time += circ_pulse_def.schedule.duration
        for q in circ_pulse_def.qubits:
            qubit_time_available[q] = stop_time

    timed_schedules = [(time, cpd.schedule) for time, cpd in zip(start_times, circ_pulse_defs)
                       if not isinstance(cpd.schedule, Barrier)]
    sched = Schedule(*timed_schedules, name=scheduled_circuit.name)
    return pad(sched)
