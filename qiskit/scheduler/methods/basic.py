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
from qiskit.pulse.channels import MeasureChannel, MemorySlot
from qiskit.pulse.commands import AcquireInstruction

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
    circuit.barrier()  # Adding a final barrier is an easy way to align the channel end times.
    sched = Schedule(name=circuit.name)

    # We schedule in reverse order to get ALAP behaviour. We need to know how far out from t=0 any
    # qubit will become occupied. We add positive shifts to these times as we go along.
    qubit_available_until = defaultdict(lambda: float("inf"))

    def update_times(inst_qubits: List[int], shift: int = 0) -> None:
        """Update the time tracker for all inst_qubits to the given time."""
        for q in inst_qubits:
            # A newly scheduled instruction on q starts at t=0 as we move backwards
            qubit_available_until[q] = 0
        for q in qubit_available_until.keys():
            if q not in inst_qubits:
                # Uninvolved qubits might be free for the duration of the new instruction
                qubit_available_until[q] += shift

    circ_pulse_defs = translate_gates_to_pulse_defs(circuit, schedule_config)
    for circ_pulse_def in reversed(circ_pulse_defs):
        if isinstance(circ_pulse_def.schedule, Barrier):
            update_times(circ_pulse_def.qubits)
        else:
            cmd_sched = circ_pulse_def.schedule
            # The new instruction should end when one of its qubits becomes occupied
            cmd_start_time = (min([qubit_available_until[q] for q in circ_pulse_def.qubits])
                              - cmd_sched.duration)
            if cmd_start_time == float("inf"):
                # These qubits haven't been used yet, so schedule the instruction at t=0
                cmd_start_time = 0
            # We have to translate qubit times forward when the cmd_start_time is negative
            shift_amount = max(0, -cmd_start_time)
            cmd_start_time = max(cmd_start_time, 0)
            sched = cmd_sched.shift(cmd_start_time).insert(shift_amount, sched, name=sched.name)
            update_times(circ_pulse_def.qubits, shift_amount)
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
    m_slots = {}  # Mapping measured qubit index to clbit
    clbit_size = int  # Storing size of classical register

    def get_measure_schedule() -> CircuitPulseDef:
        """Create a schedule to measure the qubits queued for measuring."""
        measures = set()
        all_qubits = set()
        sched = Schedule()
        for key in m_slots:
            measures.add(tuple(schedule_config.meas_map[key[0]]))
        for qubits in measures:
            all_qubits.update(qubits)
            # TODO (Issue #2704): Respect MemorySlots from the input circuit
            sched |= cmd_def.get('measure', qubits)
            # Checking if all qubits are used or not
            if all_qubits.difference(set(key[0] for key in m_slots)):
                sched = sched.exclude(channels=[MeasureChannel(q)
                                                for q in all_qubits.difference(set(key[0] for key
                                                                                   in m_slots))])
            used_m_slots = []
            for c in m_slots.values():
                used_m_slots.append(MemorySlot(c[0]))
            sched_size = len(sched.instructions)-1  # To get the AcquireInstruction position
            # Checking if all clbits are used or not
            if len(used_m_slots) != clbit_size:
                new_sched = AcquireInstruction(command=sched.instructions[sched_size][1].command,
                                               acquires=sched.instructions[sched_size][1].acquires,
                                               mem_slots=used_m_slots
                                               ) << sched.instructions[sched_size][1].start_time
                temp_sched = sched.exclude(channels=[list(set(sched.instructions
                                                              [sched_size][1].mem_slots)
                                                          .difference(set(used_m_slots)))[0]])
                sched = temp_sched.union(new_sched)
        m_slots.clear()
        return CircuitPulseDef(schedule=sched, qubits=list(all_qubits))

    for inst, qubits, clbits in circuit.data:
        inst_qubits = [qubit.index for qubit in qubits]  # We want only the indices of the qubits
        if any(q in set(key[0] for key in m_slots) for q in inst_qubits):
            # If we are operating on a qubit that was scheduled to be measured, process that first
            circ_pulse_defs.append(get_measure_schedule())
        if isinstance(inst, Barrier):
            circ_pulse_defs.append(CircuitPulseDef(schedule=inst, qubits=inst_qubits))
        elif isinstance(inst, Measure):
            m_slots[tuple(inst_qubits)] = [clbit.index for clbit in clbits]
            clbit_size = [clbit.register.size for clbit in clbits][0]
        else:
            try:
                circ_pulse_defs.append(
                    CircuitPulseDef(schedule=cmd_def.get(inst.name, inst_qubits, *inst.params),
                                    qubits=inst_qubits))
            except PulseError:
                raise QiskitError("Operation '{0}' on qubit(s) {1} not supported by the backend "
                                  "command definition. Did you remember to transpile your input "
                                  "circuit for the same backend?".format(inst.name, inst_qubits))
    if set(key[0] for key in m_slots):
        circ_pulse_defs.append(get_measure_schedule())

    return circ_pulse_defs
