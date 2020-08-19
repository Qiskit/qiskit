# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
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

from qiskit.circuit.barrier import Barrier
from qiskit.circuit.measure import Measure
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.pulse.channels import AcquireChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.macros import measure
from qiskit.scheduler.config import ScheduleConfig


CircuitPulseDef = namedtuple('CircuitPulseDef', [
    'schedule',  # The schedule which implements the quantum circuit command
    'qubits'])   # The labels of the qubits involved in the command according to the circuit


def lower_gates(circuit: QuantumCircuit, schedule_config: ScheduleConfig) -> List[CircuitPulseDef]:
    """
    Return a list of Schedules and the qubits they operate on, for each element encountered in the
``
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
        sched = measure(qubits=list(qubit_mem_slots.keys()),
                        inst_map=inst_map,
                        meas_map=schedule_config.meas_map,
                        qubit_mem_slots=qubit_mem_slots)
        qubit_mem_slots.clear()
        return CircuitPulseDef(schedule=sched,
                               qubits=[chan.index for chan in sched.channels
                                       if isinstance(chan, AcquireChannel)])

    if Measure().name in circuit.calibrations.keys():
        # TOOD
        raise NotImplementedError

    for inst, qubits, clbits in circuit.data:
        inst_qubits = [qubit.index for qubit in qubits]  # We want only the indices of the qubits

        if any(q in qubit_mem_slots for q in inst_qubits):
            # If we are operating on a qubit that was scheduled to be measured, process that first
            circ_pulse_defs.append(get_measure_schedule())

        if circuit.calibrations:
            try:
                gate_cals = circuit.calibrations[inst.name]
                schedule = gate_cals[(tuple(inst_qubits), tuple(float(p) for p in inst.params))]
                circ_pulse_defs.append(CircuitPulseDef(schedule=schedule, qubits=inst_qubits))
                continue
            except KeyError:
                pass

        if isinstance(inst, Barrier):
            circ_pulse_defs.append(CircuitPulseDef(schedule=inst, qubits=inst_qubits))
        elif isinstance(inst, Measure):
            if (len(inst_qubits) != 1 and len(clbits) != 1):
                raise QiskitError("Qubit '{}' or classical bit '{}' errored because the "
                                  "circuit Measure instruction only takes one of "
                                  "each.".format(inst_qubits, clbits))
            qubit_mem_slots[inst_qubits[0]] = clbits[0].index
        else:
            try:
                circ_pulse_defs.append(
                    CircuitPulseDef(schedule=inst_map.get(inst.name, inst_qubits, *inst.params),
                                    qubits=inst_qubits))
            except PulseError:
                raise QiskitError("Operation '{}' on qubit(s) {} not supported by the backend "
                                  "command definition. Did you remember to transpile your input "
                                  "circuit for the same backend?".format(inst.name, inst_qubits))
    if qubit_mem_slots:
        circ_pulse_defs.append(get_measure_schedule())

    return circ_pulse_defs
