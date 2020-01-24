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
Pulse utilities.
"""

from typing import List, Dict, Optional
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.schedule import Schedule
from qiskit.pulse.channels import MemorySlot
from qiskit.pulse.commands import AcquireInstruction
from qiskit.pulse.exceptions import PulseError
from qiskit.scheduler.utils import format_meas_map


def measure(qubits: List[int],
            backend: Optional['BaseBackend'] = None,
            inst_map: Optional[InstructionScheduleMap] = None,
            meas_map: Optional[List[List[int]]] = None,
            qubit_mem_slots: Optional[Dict[int, int]] = None) -> Schedule:
    """
    This function measure given qubits using OpenPulse and returns a Schedule.

    Args:
        qubits: List of qubits to be measured.
        backend: A backend instance, which contains hardware-specific data required for scheduling.
        inst_map: Mapping of circuit operations to pulse schedules. If None, defaults to the
                  ``circuit_instruction_map`` of ``backend``.
        meas_map: List of sets of qubits that must be measured together. If None, defaults to
                  the ``meas_map`` of ``backend``.
        qubit_mem_slots: Mapping of measured qubit index to classical bit index.

    Returns:
        A schedule corresponding to the inputs provided.

    Raises:
        PulseError: If both ``inst_map`` or ``meas_map``, and ``backend`` is None.
    """
    schedule = Schedule(name="Default measurement schedule for qubits {}".format(qubits))
    try:
        inst_map = inst_map or backend.defaults().circuit_instruction_map
        meas_map = meas_map or backend.configuration().meas_map
    except AttributeError:
        raise PulseError('inst_map or meas_map, and backend cannot be None simultaneously')
    if isinstance(meas_map, List):
        meas_map = format_meas_map(meas_map)

    measure_groups = set()
    for qubit in qubits:
        measure_groups.add(tuple(meas_map[qubit]))
    for measure_group_qubits in measure_groups:
        if qubit_mem_slots is not None:
            unused_mem_slots = set(measure_group_qubits) - set(qubit_mem_slots.values())
        default_sched = inst_map.get('measure', measure_group_qubits)
        for time, inst in default_sched.instructions:
            if qubit_mem_slots is not None and isinstance(inst, AcquireInstruction):
                mem_slots = []
                for channel in inst.acquires:
                    if channel.index in qubit_mem_slots.keys():
                        mem_slots.append(MemorySlot(qubit_mem_slots[channel.index]))
                    else:
                        mem_slots.append(MemorySlot(unused_mem_slots.pop()))
                new_acquire = AcquireInstruction(command=inst.command,
                                                 acquires=inst.acquires,
                                                 mem_slots=mem_slots)
                schedule = schedule.insert(time, new_acquire)
            # Measurement pulses should only be added if its qubit was measured by the user
            elif inst.channels[0].index in qubits:
                schedule = schedule.insert(time, inst)
    return schedule


def measure_all(backend: 'BaseBackend') -> Schedule:
    """
    Return a Schedule which measures all qubits of the given backend.

    Args:
        backend: A backend instance, which contains hardware-specific data required for scheduling.

    Returns:
        A schedule corresponding to the inputs provided.
    """
    return measure(qubits=list(range(backend.configuration().n_qubits)),
                   backend=backend)
