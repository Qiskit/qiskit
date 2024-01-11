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

"""Module for common pulse programming macros."""

from typing import Dict, List, Optional, Union

from qiskit.pulse import channels, exceptions, instructions, utils
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.schedule import Schedule


def measure(
    qubits: List[int],
    backend=None,
    inst_map: Optional[InstructionScheduleMap] = None,
    meas_map: Optional[Union[List[List[int]], Dict[int, List[int]]]] = None,
    qubit_mem_slots: Optional[Dict[int, int]] = None,
    measure_name: str = "measure",
) -> Schedule:
    """Return a schedule which measures the requested qubits according to the given
    instruction mapping and measure map, or by using the defaults provided by the backend.

    By default, the measurement results for each qubit are trivially mapped to the qubit
    index. This behavior is overridden by qubit_mem_slots. For instance, to measure
    qubit 0 into MemorySlot(1), qubit_mem_slots can be provided as {0: 1}.

    Args:
        qubits: List of qubits to be measured.
        backend (Union[Backend, BaseBackend]): A backend instance, which contains
            hardware-specific data required for scheduling.
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
    schedule = Schedule(name=f"Default measurement schedule for qubits {qubits}")
    try:
        inst_map = inst_map or backend.defaults().instruction_schedule_map
        meas_map = meas_map or backend.configuration().meas_map
    except AttributeError as ex:
        raise exceptions.PulseError(
            "inst_map or meas_map, and backend cannot be None simultaneously"
        ) from ex
    if isinstance(meas_map, list):
        meas_map = utils.format_meas_map(meas_map)

    measure_groups = set()
    for qubit in qubits:
        measure_groups.add(tuple(meas_map[qubit]))
    for measure_group_qubits in measure_groups:
        if qubit_mem_slots is not None:
            unused_mem_slots = set(measure_group_qubits) - set(qubit_mem_slots.values())
        try:
            default_sched = inst_map.get(measure_name, measure_group_qubits)
        except exceptions.PulseError as ex:
            raise exceptions.PulseError(
                "We could not find a default measurement schedule called '{}'. "
                "Please provide another name using the 'measure_name' keyword "
                "argument. For assistance, the instructions which are defined are: "
                "{}".format(measure_name, inst_map.instructions)
            ) from ex
        for time, inst in default_sched.instructions:
            if inst.channel.index not in qubits:
                continue
            if qubit_mem_slots and isinstance(inst, instructions.Acquire):
                if inst.channel.index in qubit_mem_slots:
                    mem_slot = channels.MemorySlot(qubit_mem_slots[inst.channel.index])
                else:
                    mem_slot = channels.MemorySlot(unused_mem_slots.pop())
                inst = instructions.Acquire(inst.duration, inst.channel, mem_slot=mem_slot)
            # Measurement pulses should only be added if its qubit was measured by the user
            schedule = schedule.insert(time, inst)

    return schedule


def measure_all(backend) -> Schedule:
    """
    Return a Schedule which measures all qubits of the given backend.

    Args:
        backend (Union[Backend, BaseBackend]): A backend instance, which contains
            hardware-specific data required for scheduling.

    Returns:
        A schedule corresponding to the inputs provided.
    """
    return measure(qubits=list(range(backend.configuration().n_qubits)), backend=backend)
