# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Assemble function for converting a list of circuits into a qobj."""
from collections import defaultdict

from qiskit.exceptions import QiskitError
from qiskit.pulse.commands import (PulseInstruction, AcquireInstruction,
                                   DelayInstruction, SamplePulse, ParametricInstruction)
from qiskit.qobj import (PulseQobj, QobjExperimentHeader,
                         PulseQobjInstruction, PulseQobjExperimentConfig,
                         PulseQobjExperiment, PulseQobjConfig, PulseLibraryItem)
from qiskit.qobj.converters import InstructionToQobjConverter, LoConfigConverter
from qiskit.qobj.converters.pulse_instruction import ParametricPulseShapes
from qiskit.qobj.utils import MeasLevel, MeasReturnType


def assemble_schedules(schedules, qobj_id, qobj_header, run_config):
    """Assembles a list of schedules into a qobj that can be run on the backend.

    Args:
        schedules (list[Schedule]): schedules to assemble
        qobj_id (int): identifier for the generated qobj
        qobj_header (QobjHeader): header to pass to the results
        run_config (RunConfig): configuration of the runtime environment
    Returns:
        PulseQobj: the Qobj to be run on the backends
    Raises:
        QiskitError: when invalid schedules or configs are provided
    """
    if hasattr(run_config, 'instruction_converter'):
        instruction_converter = run_config.instruction_converter
    else:
        instruction_converter = InstructionToQobjConverter

    qobj_config = run_config.to_dict()

    qubit_lo_freq = qobj_config.get('qubit_lo_freq', None)
    if qubit_lo_freq is None:
        raise QiskitError('qubit_lo_freq must be supplied.')

    meas_lo_freq = qobj_config.get('meas_lo_freq', None)
    if meas_lo_freq is None:
        raise QiskitError('meas_lo_freq must be supplied.')

    meas_map = qobj_config.pop('meas_map', None)

    # convert enums to serialized values
    meas_return = qobj_config.get('meas_return', 'avg')
    if isinstance(meas_return, MeasReturnType):
        qobj_config['meas_return'] = meas_return.value

    meas_level = qobj_config.get('meas_return', 2)
    if isinstance(meas_level, MeasLevel):
        qobj_config['meas_level'] = meas_level.value

    instruction_converter = instruction_converter(PulseQobjInstruction, **qobj_config)

    qubit_lo_range = qobj_config.pop('qubit_lo_range', None)
    meas_lo_range = qobj_config.pop('meas_lo_range', None)
    lo_converter = LoConfigConverter(PulseQobjExperimentConfig,
                                     qubit_lo_range=qubit_lo_range,
                                     meas_lo_range=meas_lo_range,
                                     **qobj_config)

    memory_slot_size = 0

    # Pack everything into the Qobj
    qobj_schedules = []
    user_pulselib = {}
    for idx, schedule in enumerate(schedules):
        # instructions
        max_memory_slot = 0
        qobj_instructions = []
        acquire_instruction_map = defaultdict(list)

        # Instructions are returned as tuple of shifted time and instruction
        for shift, instruction in schedule.instructions:
            # TODO: support conditional gate

            if isinstance(instruction, ParametricInstruction):
                pulse_shape = ParametricPulseShapes(type(instruction.command)).name
                if pulse_shape not in run_config.parametric_pulses:
                    # Convert to SamplePulse if the backend does not support it
                    instruction = PulseInstruction(instruction.command.get_sample_pulse(),
                                                   instruction.channels[0],
                                                   name=instruction.name)

            if isinstance(instruction, PulseInstruction):
                name = instruction.command.name
                if name in user_pulselib and instruction.command != user_pulselib[name]:
                    name = "{0}-{1:x}".format(name, hash(instruction.command.samples.tostring()))
                    instruction = PulseInstruction(
                        command=SamplePulse(name=name, samples=instruction.command.samples),
                        name=instruction.name,
                        channel=instruction.channels[0])
                # add samples to pulse library
                user_pulselib[name] = instruction.command

            if isinstance(instruction, AcquireInstruction):
                max_memory_slot = max(max_memory_slot,
                                      *[slot.index for slot in instruction.mem_slots])
                # Acquires have a single AcquireChannel per inst, but we have to bundle them
                # together into the Qobj as one instruction with many channels
                acquire_instruction_map[(shift, instruction.command)].append(instruction)
                continue

            if isinstance(instruction, DelayInstruction):
                # delay instructions are ignored as timing is explicit within qobj
                continue

            qobj_instructions.append(instruction_converter(shift, instruction))

        if acquire_instruction_map:
            if meas_map:
                _validate_meas_map(acquire_instruction_map, meas_map)
            for (shift, _), instructions in acquire_instruction_map.items():
                qubits, mem_slots, reg_slots = _bundle_channel_indices(instructions)
                qobj_instructions.append(
                    instruction_converter.convert_single_acquires(
                        shift, instructions[0],
                        qubits=qubits, memory_slot=mem_slots, register_slot=reg_slots))

        # memory slot size is memory slot index + 1 because index starts from zero
        exp_memory_slot_size = max_memory_slot + 1
        memory_slot_size = max(memory_slot_size, exp_memory_slot_size)

        # experiment header
        # TODO: add other experimental header items (see circuit assembler)
        qobj_experiment_header = QobjExperimentHeader(
            memory_slots=exp_memory_slot_size,
            name=schedule.name or 'Experiment-%d' % idx
        )

        qobj_schedules.append({
            'header': qobj_experiment_header,
            'instructions': qobj_instructions
        })

    # set number of memoryslots
    qobj_config['memory_slots'] = memory_slot_size

    # setup pulse_library
    qobj_config['pulse_library'] = [PulseLibraryItem(name=pulse.name, samples=pulse.samples)
                                    for pulse in user_pulselib.values()]

    # convert lo frequencies to GHz
    qobj_config['qubit_lo_freq'] = [freq/1e9 for freq in qubit_lo_freq]
    qobj_config['meas_lo_freq'] = [freq/1e9 for freq in meas_lo_freq]

    # create qobj experiment field
    experiments = []
    schedule_los = qobj_config.pop('schedule_los', [])

    if len(schedule_los) == 1:
        lo_dict = schedule_los[0]
        # update global config
        q_los = lo_converter.get_qubit_los(lo_dict)
        if q_los:
            qobj_config['qubit_lo_freq'] = [freq/1e9 for freq in q_los]
        m_los = lo_converter.get_meas_los(lo_dict)
        if m_los:
            qobj_config['meas_lo_freq'] = [freq/1e9 for freq in m_los]

    if schedule_los:
        # multiple frequency setups
        if len(qobj_schedules) == 1:
            # frequency sweep
            for lo_dict in schedule_los:
                experiments.append(PulseQobjExperiment(
                    instructions=qobj_schedules[0]['instructions'],
                    header=qobj_schedules[0]['header'],
                    config=lo_converter(lo_dict)
                ))
        elif len(qobj_schedules) == len(schedule_los):
            # n:n setup
            for lo_dict, schedule in zip(schedule_los, qobj_schedules):
                experiments.append(PulseQobjExperiment(
                    instructions=schedule['instructions'],
                    header=schedule['header'],
                    config=lo_converter(lo_dict)
                ))
        else:
            raise QiskitError('Invalid LO setting is specified. '
                              'The LO should be configured for each schedule, or '
                              'single setup for all schedules (unique), or '
                              'multiple setups for a single schedule (frequency sweep),'
                              'or no LO configured at all.')
    else:
        # unique frequency setup
        for schedule in qobj_schedules:
            experiments.append(PulseQobjExperiment(
                instructions=schedule['instructions'],
                header=schedule['header'],
            ))

    qobj_config = PulseQobjConfig(**qobj_config)

    return PulseQobj(qobj_id=qobj_id,
                     config=qobj_config,
                     experiments=experiments,
                     header=qobj_header)


def _validate_meas_map(instruction_map, meas_map):
    """Validate all qubits tied in meas_map are to be acquired."""
    meas_map_sets = [set(m) for m in meas_map]

    # Check each acquisition time individually
    for _, instructions in instruction_map.items():

        measured_qubits = set()
        for inst in instructions:
            measured_qubits.update([acq.index for acq in inst.acquires])

        for meas_set in meas_map_sets:
            intersection = measured_qubits.intersection(meas_set)
            if intersection and intersection != meas_set:
                raise QiskitError('Qubits to be acquired: {0} do not satisfy required qubits '
                                  'in measurement map: {1}'.format(measured_qubits, meas_set))


def _bundle_channel_indices(instructions):
    """From the list of AcquireInstructions, bundle the indices of the acquire channels,
    memory slots, and register slots into a 3-tuple of lists."""
    qubits = []
    mem_slots = []
    reg_slots = []
    for inst in instructions:
        qubits.extend(aq.index for aq in inst.acquires)
        mem_slots.extend(mem_slot.index for mem_slot in inst.mem_slots)
        reg_slots.extend(reg.index for reg in inst.reg_slots)
    return qubits, mem_slots, reg_slots
