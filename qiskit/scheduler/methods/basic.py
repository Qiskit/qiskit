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

"""The most straightforward scheduling methods: scheduling as early or as late as possible.

Warning: Currently for both of these methods, the MemorySlots in circuit Measures are ignored.
Qubits will be measured into the MemorySlot which matches the measured qubit's index. (Issue #2704)
"""

from collections import defaultdict, namedtuple
from typing import List

from qiskit.circuit.measure import Measure
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.extensions.standard.barrier import Barrier
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule

from qiskit.scheduler.config import ScheduleConfig


CircuitPulseDef = namedtuple('CircuitPulseDef', [
    'schedule',  # The schedule which implements the quantum circuit command
    'qubits'])   # The labels of the qubits involved in the command according to the circuit


def as_soon_as_possible(circuit: QuantumCircuit,
                        schedule_config: ScheduleConfig) -> Schedule:
    """
    Return the pulse Schedule which implements the input circuit using an "as soon as possible"
    (asap) scheduling policy. Circuit instructions are first each mapped to equivalent pulse
    Schedules according to the command definition given by the schedule_config. Then, this circuit
    instruction-equivalent Schedule is appended at the earliest time at which all qubits involved
    in the instruction are available.

    Args:
        circuit: The quantum circuit to translate
        schedule_config: Backend specific parameters used for building the Schedule
    Returns:
        A schedule corresponding to the input `circuit` with pulses occurring as early as possible
    """
    sched = Schedule(name=circuit.name)

    qubit_time_available = defaultdict(int)

    def update_times(inst_qubits: List[int], time: int = 0) -> None:
        """Update the time tracker for all inst_qubits to the given time."""
        for q in inst_qubits:
            qubit_time_available[q] = time

    circ_pulse_defs = translate_gates_to_pulse_defs(circuit, schedule_config)
    for circ_pulse_def in circ_pulse_defs:
        time = max(qubit_time_available[q] for q in circ_pulse_def.qubits)
        if isinstance(circ_pulse_def.schedule, Barrier):
            update_times(circ_pulse_def.qubits, time)
        else:
            sched = sched.insert(time, circ_pulse_def.schedule)
            update_times(circ_pulse_def.qubits, time + circ_pulse_def.schedule.duration)
    return sched


def as_late_as_possible(circuit: QuantumCircuit,
                        schedule_config: ScheduleConfig) -> Schedule:
    """
    Return the pulse Schedule which implements the input circuit using an "as late as possible"
    (alap) scheduling policy. Circuit instructions are first each mapped to equivalent pulse
    Schedules according to the command definition given by the schedule_config. Then, this circuit
    instruction-equivalent Schedule is appended at the latest time that it can be without allowing
    unnecessary time between instructions or allowing instructions with common qubits to overlap.

    This method should improves the outcome fidelity over ASAP scheduling, because we may
    maximize the time that the qubit remains in the ground state.

    Args:
        circuit: The quantum circuit to translate
        schedule_config: Backend specific parameters used for building the Schedule
    Returns:
        A schedule corresponding to the input `circuit` with pulses occurring as late as possible
    """
    sched = Schedule(name=circuit.name)
    # Align channel end times.
    circuit.barrier()
    # We schedule in reverse order to get ALAP behaviour. We need to know how far out from t=0 any
    # qubit will become occupied. We add positive shifts to these times as we go along.
    # The time is initialized to 0 because all qubits are involved in the final barrier.
    qubit_available_until = defaultdict(lambda: 0)

    def update_times(inst_qubits: List[int], shift: int = 0, cmd_start_time: int = 0) -> None:
        """Update the time tracker for all inst_qubits to the given time."""
        for q in inst_qubits:
            qubit_available_until[q] = cmd_start_time
        for q in qubit_available_until.keys():
            if q not in inst_qubits:
                # Uninvolved qubits might be free for the duration of the new instruction
                qubit_available_until[q] += shift

    circ_pulse_defs = translate_gates_to_pulse_defs(circuit, schedule_config)
    for circ_pulse_def in reversed(circ_pulse_defs):
        cmd_sched = circ_pulse_def.schedule
        # The new instruction should end when one of its qubits becomes occupied
        cmd_start_time = (min([qubit_available_until[q] for q in circ_pulse_def.qubits])
                          - getattr(cmd_sched, 'duration', 0))  # Barrier has no duration
        # We have to translate qubit times forward when the cmd_start_time is negative
        shift_amount = max(0, -cmd_start_time)
        cmd_start_time = max(cmd_start_time, 0)
        if not isinstance(circ_pulse_def.schedule, Barrier):
            sched = cmd_sched.shift(cmd_start_time).insert(shift_amount, sched, name=sched.name)
        update_times(circ_pulse_def.qubits, shift_amount, cmd_start_time)
    return sched


def translate_gates_to_pulse_defs(circuit: QuantumCircuit,
                                  schedule_config: ScheduleConfig) -> List[CircuitPulseDef]:
    """
    Without concern for the final schedule, extract and return a list of Schedules and the qubits
    they operate on, for each element encountered in the input circuit. Measures are grouped when
    possible, so qc.measure(q0, c0)/qc.measure(q1, c1) will generate a synchronous measurement
    pulse.

    Args:
        circuit: The quantum circuit to translate
        schedule_config: Backend specific parameters used for building the Schedule
    Returns:
        A list of CircuitPulseDefs: the pulse definition for each circuit element
    Raises:
        QiskitError: If circuit uses a command that isn't defined in config.cmd_def
    """
    circ_pulse_defs = []

    cmd_def = schedule_config.cmd_def
    measured_qubits = set()  # Collect qubits that would like to be measured

    def get_measure_schedule() -> CircuitPulseDef:
        """Create a schedule to measure the qubits queued for measuring."""
        measures = set()
        all_qubits = set()
        sched = Schedule()
        for q in measured_qubits:
            measures.add(tuple(schedule_config.meas_map[q]))
        for qubits in measures:
            all_qubits.update(qubits)
            # TODO (Issue #2704): Respect MemorySlots from the input circuit
            sched |= cmd_def.get('measure', qubits)
        measured_qubits.clear()
        return CircuitPulseDef(schedule=sched, qubits=list(all_qubits))

    for inst, qubits, _ in circuit.data:
        inst_qubits = [qubit.index for qubit in qubits]  # We want only the indices of the qubits
        if any(q in measured_qubits for q in inst_qubits):
            # If we are operating on a qubit that was scheduled to be measured, process that first
            circ_pulse_defs.append(get_measure_schedule())
        if isinstance(inst, Barrier):
            circ_pulse_defs.append(CircuitPulseDef(schedule=inst, qubits=inst_qubits))
        elif isinstance(inst, Measure):
            measured_qubits.update(inst_qubits)
        else:
            try:
                circ_pulse_defs.append(
                    CircuitPulseDef(schedule=cmd_def.get(inst.name, inst_qubits, *inst.params),
                                    qubits=inst_qubits))
            except PulseError:
                raise QiskitError("Operation '{0}' on qubit(s) {1} not supported by the backend "
                                  "command definition. Did you remember to transpile your input "
                                  "circuit for the same backend?".format(inst.name, inst_qubits))
    if measured_qubits:
        circ_pulse_defs.append(get_measure_schedule())

    return circ_pulse_defs
