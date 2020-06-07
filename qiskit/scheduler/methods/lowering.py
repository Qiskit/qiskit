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

"""Lower gates to schedules. The relative timing within gates is respected. This
module handles the translation, but does not handle timing.
"""
from collections import namedtuple
from typing import List

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.barrier import Barrier
from qiskit.circuit.delay import Delay
from qiskit.circuit.measure import Measure
from qiskit.exceptions import QiskitError
from qiskit.pulse.channels import AcquireChannel, DriveChannel, MeasureChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule
from qiskit.pulse import instructions as pulse_inst

from qiskit.scheduler.config import ScheduleConfig
from qiskit.scheduler.utils import measure


CircuitPulseDef = namedtuple('CircuitPulseDef', [
    'schedule',  # The schedule which implements the quantum circuit command
    'qubits'])   # The labels of the qubits involved in the command according to the circuit


def lower_gates(circuit: QuantumCircuit, schedule_config: ScheduleConfig) -> List[CircuitPulseDef]:
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

        elif isinstance(inst, Delay):
            if inst.unit == 'dt':
                duration = inst.duration
            else:
                duration_s = _convert_to_seconds(inst.duration, inst.unit)
                duration = int(duration_s // schedule_config.dt)
            sched = Schedule(name=inst.name)
            for qubit in inst_qubits:
                for channel in [DriveChannel, AcquireChannel, MeasureChannel]:
                    sched += pulse_inst.Delay(duration=duration, channel=channel(qubit))
            circ_pulse_defs.append(CircuitPulseDef(schedule=sched, qubits=inst_qubits))

        elif isinstance(inst, Measure):
            if (len(inst_qubits) != 1 and len(clbits) != 1):
                raise QiskitError("Qubit '{0}' or classical bit '{1}' errored because the "
                                  "circuit Measure instruction only takes one of "
                                  "each.".format(inst_qubits, clbits))
            if clbits:
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


def _convert_to_seconds(value: float, unit: str) -> float:
    """Convert input value to seconds by applying unit to value."""
    prefixes = {'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'µ': 1e-6,
                'm': 1e-3, 'k': 1e3, 'M': 1e6, 'G': 1e9, 's': 1}
    if not unit:
        return value
    try:
        return value * prefixes[unit[0]]
    except KeyError:
        raise QiskitError("Error parsing delay operation duration.")
