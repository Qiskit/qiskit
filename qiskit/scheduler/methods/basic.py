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
The most straightforward scheduling methods: scheduling **as early** or **as late** as possible.
"""
from collections import defaultdict
from typing import List

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.barrier import Barrier
from qiskit.pulse.schedule import Schedule

from qiskit.scheduler.config import ScheduleConfig
from qiskit.scheduler.lowering import lower_gates


def as_soon_as_possible(circuit: QuantumCircuit,
                        schedule_config: ScheduleConfig) -> Schedule:
    """
    Return the pulse Schedule which implements the input circuit using an "as soon as possible"
    (asap) scheduling policy.

    Circuit instructions are first each mapped to equivalent pulse
    Schedules according to the command definition given by the schedule_config. Then, this circuit
    instruction-equivalent Schedule is appended at the earliest time at which all qubits involved
    in the instruction are available.

    Args:
        circuit: The quantum circuit to translate.
        schedule_config: Backend specific parameters used for building the Schedule.

    Returns:
        A schedule corresponding to the input ``circuit`` with pulses occurring as early as
        possible.
    """
    qubit_time_available = defaultdict(int)

    def update_times(inst_qubits: List[int], time: int = 0) -> None:
        """Update the time tracker for all inst_qubits to the given time."""
        for q in inst_qubits:
            qubit_time_available[q] = time

    start_times = []
    circ_pulse_defs = lower_gates(circuit, schedule_config)
    for circ_pulse_def in circ_pulse_defs:
        start_time = max(qubit_time_available[q] for q in circ_pulse_def.qubits)
        stop_time = start_time
        if not isinstance(circ_pulse_def.schedule, Barrier):
            stop_time += circ_pulse_def.schedule.duration

        start_times.append(start_time)
        update_times(circ_pulse_def.qubits, stop_time)

    timed_schedules = [(time, cpd.schedule) for time, cpd in zip(start_times, circ_pulse_defs)
                       if not isinstance(cpd.schedule, Barrier)]
    return Schedule(*timed_schedules, name=circuit.name)


def as_late_as_possible(circuit: QuantumCircuit,
                        schedule_config: ScheduleConfig) -> Schedule:
    """
    Return the pulse Schedule which implements the input circuit using an "as late as possible"
    (alap) scheduling policy.

    Circuit instructions are first each mapped to equivalent pulse
    Schedules according to the command definition given by the schedule_config. Then, this circuit
    instruction-equivalent Schedule is appended at the latest time that it can be without allowing
    unnecessary time between instructions or allowing instructions with common qubits to overlap.

    This method should improves the outcome fidelity over ASAP scheduling, because we may
    maximize the time that the qubit remains in the ground state.

    Args:
        circuit: The quantum circuit to translate.
        schedule_config: Backend specific parameters used for building the Schedule.

    Returns:
        A schedule corresponding to the input ``circuit`` with pulses occurring as late as
        possible.
    """
    qubit_time_available = defaultdict(int)

    def update_times(inst_qubits: List[int], time: int = 0) -> None:
        """Update the time tracker for all inst_qubits to the given time."""
        for q in inst_qubits:
            qubit_time_available[q] = time

    rev_stop_times = []
    circ_pulse_defs = lower_gates(circuit, schedule_config)
    for circ_pulse_def in reversed(circ_pulse_defs):
        start_time = max(qubit_time_available[q] for q in circ_pulse_def.qubits)
        stop_time = start_time
        if not isinstance(circ_pulse_def.schedule, Barrier):
            stop_time += circ_pulse_def.schedule.duration

        rev_stop_times.append(stop_time)
        update_times(circ_pulse_def.qubits, stop_time)

    last_stop = max(t for t in qubit_time_available.values()) if qubit_time_available else 0
    start_times = [last_stop - t for t in reversed(rev_stop_times)]
    timed_schedules = [(time, cpd.schedule) for time, cpd in zip(start_times, circ_pulse_defs)
                       if not isinstance(cpd.schedule, Barrier)]
    return Schedule(*timed_schedules, name=circuit.name)
