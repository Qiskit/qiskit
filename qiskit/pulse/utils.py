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
import warnings
from typing import List, Dict, Optional
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.schedule import Schedule
from qiskit.pulse.channels import MemorySlot
from qiskit.pulse.commands import AcquireInstruction
# pylint: disable=unused-argument


def align_measures(schedules, cmd_def, cal_gate, max_calibration_duration=None, align_time=None):
    """
    This function has been moved!
    """
    warnings.warn("The function `align_measures` has been moved to qiskit.pulse.reschedule. "
                  "It cannot be invoked from `utils` anymore (this call returns None).")


def add_implicit_acquires(schedule, meas_map):
    """
    This function has been moved!
    """
    warnings.warn("The function `add_implicit_acquires` has been moved to qiskit.pulse.reschedule."
                  " It cannot be invoked from `utils` anymore (this call returns None).")


def pad(schedule, channels=None, until=None):
    """
    This function has been moved!
    """
    warnings.warn("The function `pad` has been moved to qiskit.pulse.reschedule. It cannot be "
                  "invoked from `utils` anymore (this call returns None).")


def measure(qubits: List[int],
            schedule: Schedule,
            inst_map: Optional[InstructionScheduleMap],
            meas_map: List[List[int]],
            backend: Optional['BaseBackend'] = None,
            qubit_mem_slots: Optional[Dict[int, int]] = None) -> Schedule:
    """
    This is a utility function to measure qubits using OpenPulse.

    Args:
        qubits: List of qubits to be measured.
        schedule: Schedule of the circuit.
        inst_map: Mapping of circuit operations to pulse schedules. If None, defaults to the
                  ``circuit_instruction_map`` of ``backend``.
        meas_map: List of sets of qubits that must be measured together. If None, defaults to
                  the ``meas_map`` of ``backend``.
        backend: A backend instance, which contains hardware-specific data required for scheduling.
        qubit_mem_slots: Mapping of measured qubit index to classical bit index.

    Returns:
        A schedule corresponding to the inputs provided.
    """

    inst_map = inst_map or backend.defaults().circuit_instruction_map
    meas_map = meas_map or backend.configuration().meas_map
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
