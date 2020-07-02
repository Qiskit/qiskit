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
import warnings
from collections import defaultdict

from qiskit.circuit.barrier import Barrier
from qiskit.circuit.measure import Measure
from qiskit.circuit.quantumcircuit import QuantumCircuit
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
    """
    # trace start times
    qubit_time_available = defaultdict(int)
    start_times = []
    measure_times = []
    for inst, qubits, _ in scheduled_circuit.data:
        start_time = qubit_time_available[qubits[0]]
        if isinstance(inst, Measure):
            measure_times.append(start_time)
        else:
            start_times.append(start_time)
        for q in qubits:
            qubit_time_available[q] += inst.duration

    if measure_times:
        measure_time = measure_times[0]
        for time in measure_times:
            if time != measure_time:
                # TODO: should we raise an exception?
                warnings.warn('Not all measurements are done at once at the last.'
                              'Resulting schedule may be incorrect.',
                              UserWarning)
        start_times.append(measure_time)

    circ_pulse_defs = lower_gates(scheduled_circuit, schedule_config)
    assert len(start_times) == len(circ_pulse_defs)
    timed_schedules = [(time, cpd.schedule) for time, cpd in zip(start_times, circ_pulse_defs)
                       if not isinstance(cpd.schedule, Barrier)]
    sched = Schedule(*timed_schedules, name=scheduled_circuit.name)
    assert sched.duration == scheduled_circuit.duration
    return pad(sched)
