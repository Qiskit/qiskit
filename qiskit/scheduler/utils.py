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

"""Scheduling utility functions."""

from typing import Dict, List, Optional, Union
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.schedule import Schedule
from qiskit.pulse.channels import MemorySlot
from qiskit.pulse.commands import AcquireInstruction
from qiskit.pulse.exceptions import PulseError


def format_meas_map(meas_map: List[List[int]]) -> Dict[int, List[int]]:
    """
    Return a mapping from qubit label to measurement group given the nested list meas_map returned
    by a backend configuration. (Qubits can not always be measured independently.) Sorts the
    measurement group for consistency.

    Args:
        meas_map: Groups of qubits that get measured together, for example: [[0, 1], [2, 3, 4]]
    Returns:
        Measure map in map format
    """
    qubit_mapping = {}
    for sublist in meas_map:
        sublist.sort()
        for q in sublist:
            qubit_mapping[q] = sublist
    return qubit_mapping


def measure(qubits: List[int],
            backend: Optional['BaseBackend'] = None,
            inst_map: Optional[InstructionScheduleMap] = None,
            meas_map: Optional[Union[List[List[int]], Dict[int, List[int]]]] = None,
            qubit_mem_slots: Optional[Dict[int, int]] = None,
            measure_name: str = 'measure') -> Schedule:
    """
    Return a schedule which measures the requested qubits according to the given
    instruction mapping and measure map, or by using the defaults provided by the backend.

    By default, the measurement results for each qubit are trivially mapped to the qubit
    index. This behavior is overridden by qubit_mem_slots. For instance, to measure
    qubit 0 into MemorySlot(1), qubit_mem_slots can be provided as {0: 1}.

    Args:
        qubits: List of qubits to be measured.
        backend: A backend instance, which contains hardware-specific data required for scheduling.
        inst_map: Mapping of circuit operations to pulse schedules. If None, defaults to the
                  ``instruction_schedule_map`` of ``backend``.
        meas_map: List of sets of qubits that must be measured together. If None, defaults to
                  the ``meas_map`` of ``backend``.
        qubit_mem_slots: Mapping of measured qubit index to classical bit index.
        measure_name: Name of the measurement schedule.

    Returns:
        A measurement schedule corresponding to the inputs provided.

    Raises:
        PulseError: If both ``inst_map`` or ``meas_map``, and ``backend`` is None.
    """
    schedule = Schedule(name="Default measurement schedule for qubits {}".format(qubits))
    try:
        inst_map = inst_map or backend.defaults().instruction_schedule_map
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
        try:
            default_sched = inst_map.get(measure_name, measure_group_qubits)
        except PulseError:
            raise PulseError("We could not find a default measurement schedule called '{}'. "
                             "Please provide another name using the 'measure_name' keyword "
                             "argument. For assistance, the instructions which are defined are: "
                             "{}".format(measure_name, inst_map.instructions))

        for time, inst in default_sched.instructions:
            if qubit_mem_slots and isinstance(inst, AcquireInstruction):
                for channel in inst.acquires:
                    if channel.index in qubit_mem_slots:
                        mem_slot = MemorySlot(qubit_mem_slots[channel.index])
                    else:
                        mem_slot = MemorySlot(unused_mem_slots.pop())
                    schedule = schedule.insert(time, AcquireInstruction(command=inst.command,
                                                                        acquire=channel,
                                                                        mem_slot=mem_slot))
            elif qubit_mem_slots is None and isinstance(inst, AcquireInstruction):
                schedule = schedule.insert(time, inst)
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
