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
The most straightforward scheduling methods: scheduling **as early** or **as late** as possible.
"""

from collections import defaultdict, namedtuple
from typing import List

from qiskit.circuit.measure import Measure
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.circuit.barrier import Barrier
from qiskit.pulse.channels import AcquireChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.macros import measure
from qiskit.pulse.schedule import Schedule

from qiskit.scheduler.config import ScheduleConfig


CircuitPulseDef = namedtuple('CircuitPulseDef', [
    'schedule',  # The schedule which implements the quantum circuit command
    'qubits'])   # The labels of the qubits involved in the command according to the circuit


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
    circ_pulse_defs = translate_gates_to_pulse_defs(circuit, schedule_config)
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
    circ_pulse_defs = translate_gates_to_pulse_defs(circuit, schedule_config)
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


def translate_gates_to_pulse_defs(circuit: QuantumCircuit,
                                  schedule_config: ScheduleConfig) -> List[CircuitPulseDef]:
    """
    Return a list of Schedules and the qubits they operate on, for each element encountered in th
    input circuit.

    Without concern for the final schedule, extract and return a list of Schedules and the qubits
    they operate on, for each element encountered in the input circuit. Measures are grouped when
    possible, so ``qc.measure(q0, c0)`` or ``qc.measure(q1, c1)`` will generate a synchronous
    measurement pulse.

    Args:
        circuit: The quantum circuit to translate.
        schedule_config: Backend specific parameters used for building the Schedule.

    Returns:
        A list of CircuitPulseDefs: the pulse definition for each circuit element.

    Raises:
        QiskitError: If circuit uses a command that isn't defined in config.inst_map.
    """
    circ_pulse_defs = []

    inst_map = schedule_config.inst_map
    qubit_mem_slots = {}  # Map measured qubit index to classical bit index

    def get_measure_schedule() -> CircuitPulseDef:
        """Create a schedule to measure the qubits queued for measuring."""
        sched = Schedule()
        sched += measure(qubits=list(qubit_mem_slots.keys()),
                         inst_map=inst_map,
                         meas_map=schedule_config.meas_map,
                         qubit_mem_slots=qubit_mem_slots)
        qubit_mem_slots.clear()
        return CircuitPulseDef(schedule=sched,
                               qubits=[chan.index for chan in sched.channels
                                       if isinstance(chan, AcquireChannel)])

    for inst, qubits, clbits in circuit.data:
        inst_qubits = [qubit.index for qubit in qubits]  # We want only the indices of the qubits
        if any(q in qubit_mem_slots for q in inst_qubits):
            # If we are operating on a qubit that was scheduled to be measured, process that first
            circ_pulse_defs.append(get_measure_schedule())
        if isinstance(inst, Barrier):
            circ_pulse_defs.append(CircuitPulseDef(schedule=inst, qubits=inst_qubits))
        elif isinstance(inst, Measure):
            if (len(inst_qubits) != 1 and len(clbits) != 1):
                raise QiskitError("Qubit '{0}' or classical bit '{1}' errored because the "
                                  "circuit Measure instruction only takes one of "
                                  "each.".format(inst_qubits, clbits))
            qubit_mem_slots[inst_qubits[0]] = clbits[0].index
        else:
            try:
                circ_pulse_defs.append(
                    CircuitPulseDef(schedule=inst_map.get(inst.name, inst_qubits, *inst.params),
                                    qubits=inst_qubits))
            except PulseError:
                raise QiskitError("Operation '{0}' on qubit(s) {1} not supported by the backend "
                                  "command definition. Did you remember to transpile your input "
                                  "circuit for the same backend?".format(inst.name, inst_qubits))
    if qubit_mem_slots:
        circ_pulse_defs.append(get_measure_schedule())

    return circ_pulse_defs
