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
from typing import Any, Dict, List, Tuple
from collections import defaultdict

from qiskit.exceptions import QiskitError
from qiskit.pulse import Schedule
from qiskit.pulse.commands import (Command, PulseInstruction, Acquire, AcquireInstruction,
                                   DelayInstruction, SamplePulse, ParametricInstruction)
from qiskit.qobj import (PulseQobj, QobjHeader, QobjExperimentHeader,
                         PulseQobjInstruction, PulseQobjExperimentConfig,
                         PulseQobjExperiment, PulseQobjConfig, PulseLibraryItem)
from qiskit.qobj.converters import InstructionToQobjConverter, LoConfigConverter
from qiskit.qobj.converters.pulse_instruction import ParametricPulseShapes
from qiskit.qobj.utils import MeasLevel, MeasReturnType

from .run_config import RunConfig


def assemble_schedules(schedules: List[Schedule],
                       qobj_id: int,
                       qobj_header: QobjHeader,
                       run_config: RunConfig) -> PulseQobj:
    """Assembles a list of schedules into a qobj that can be run on the backend.

    Args:
        schedules: Schedules to assemble.
        qobj_id: Identifier for the generated qobj.
        qobj_header: Header to pass to the results.
        run_config: Configuration of the runtime environment.

    Returns:
        The Qobj to be run on the backends.

    Raises:
        QiskitError: when frequency settings are not supplied.
    """
    if not hasattr(run_config, 'qubit_lo_freq'):
        raise QiskitError('qubit_lo_freq must be supplied.')
    if not hasattr(run_config, 'meas_lo_freq'):
        raise QiskitError('meas_lo_freq must be supplied.')

    lo_converter = LoConfigConverter(PulseQobjExperimentConfig,
                                     **run_config.to_dict())
    experiments, experiment_config = _assemble_experiments(schedules,
                                                           lo_converter,
                                                           run_config)
    qobj_config = _assemble_config(lo_converter, experiment_config, run_config)

    return PulseQobj(experiments=experiments,
                     qobj_id=qobj_id,
                     header=qobj_header,
                     config=qobj_config)


def _assemble_experiments(
        schedules: List[Schedule],
        lo_converter: LoConfigConverter,
        run_config: RunConfig
) -> Tuple[List[PulseQobjExperiment], Dict[str, Any]]:
    """Assembles a list of schedules into PulseQobjExperiments, and returns related metadata that
    will be assembled into the Qobj configuration.

    Args:
        schedules: Schedules to assemble.
        lo_converter: The configured frequency converter and validator.
        run_config: Configuration of the runtime environment.

    Returns:
        The list of assembled experiments, and the dictionary of related experiment config.

    Raises:
        QiskitError: when frequency settings are not compatible with the experiments.
    """
    freq_configs = [lo_converter(lo_dict) for lo_dict in getattr(run_config, 'schedule_los', [])]

    if len(schedules) > 1 and len(freq_configs) not in [0, 1, len(schedules)]:
        raise QiskitError('Invalid frequency setting is specified. If the frequency is specified, '
                          'it should be configured the same for all schedules, configured for each '
                          'schedule, or a list of frequencies should be provided for a single '
                          'frequency sweep schedule.')

    instruction_converter = getattr(run_config, 'instruction_converter', InstructionToQobjConverter)
    instruction_converter = instruction_converter(PulseQobjInstruction, **run_config.to_dict())

    user_pulselib = {}
    experiments = []
    for idx, schedule in enumerate(schedules):
        qobj_instructions, user_pulses, max_memory_slot = _assemble_instructions(
            schedule,
            instruction_converter,
            run_config)
        user_pulselib.update(user_pulses)

        # TODO: add other experimental header items (see circuit assembler)
        qobj_experiment_header = QobjExperimentHeader(
            memory_slots=max_memory_slot + 1,  # Memory slots are 0 indexed
            name=schedule.name or 'Experiment-%d' % idx)

        experiment = PulseQobjExperiment(
            header=qobj_experiment_header,
            instructions=qobj_instructions)
        if freq_configs:
            # This handles the cases where one frequency setting applies to all experiments and
            # where each experiment has a different frequency
            freq_idx = idx if len(freq_configs) != 1 else 0
            experiment.config = freq_configs[freq_idx]

        experiments.append(experiment)

    # Frequency sweep
    if freq_configs and len(experiments) == 1:
        experiment = experiments[0]
        experiments = []
        for freq_config in freq_configs:
            experiments.append(PulseQobjExperiment(
                header=experiment.header,
                instructions=experiment.instructions,
                config=freq_config))

    # Top level Qobj configuration
    experiment_config = {
        'pulse_library': [PulseLibraryItem(name=pulse.name, samples=pulse.samples)
                          for pulse in user_pulselib.values()],
        'memory_slots': max([exp.header.memory_slots for exp in experiments])
    }

    return experiments, experiment_config


def _assemble_instructions(
        schedule: Schedule,
        instruction_converter: InstructionToQobjConverter,
        run_config: RunConfig
) -> Tuple[List[PulseQobjInstruction], Dict[str, Command], int]:
    """Assembles the instructions in a schedule into a list of PulseQobjInstructions and returns
    related metadata that will be assembled into the Qobj configuration.

    Args:
        schedule: Schedule to assemble.
        instruction_converter: A converter instance which can convert PulseInstructions to
                               PulseQobjInstructions.
        run_config: Configuration of the runtime environment.

    Returns:
        A list of converted instructions, the user pulse library dictionary (from pulse name to
        pulse command), and the maximum number of readout memory slots used by this Schedule.
    """
    max_memory_slot = 0
    qobj_instructions = []
    user_pulselib = {}

    acquire_instruction_map = defaultdict(list)
    for time, instruction in schedule.instructions:

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
            acquire_instruction_map[(time, instruction.command)].append(instruction)
            continue

        if isinstance(instruction, DelayInstruction):
            # delay instructions are ignored as timing is explicit within qobj
            continue

        qobj_instructions.append(instruction_converter(time, instruction))

    if acquire_instruction_map:
        if hasattr(run_config, 'meas_map'):
            _validate_meas_map(acquire_instruction_map, run_config.meas_map)
        for (time, _), instructions in acquire_instruction_map.items():
            qubits, mem_slots, reg_slots = _bundle_channel_indices(instructions)
            qobj_instructions.append(
                instruction_converter.convert_single_acquires(
                    time, instructions[0],
                    qubits=qubits, memory_slot=mem_slots, register_slot=reg_slots))

    return qobj_instructions, user_pulselib, max_memory_slot


def _validate_meas_map(instruction_map: Dict[Tuple[int, Acquire], List[AcquireInstruction]],
                       meas_map: List[List[int]]) -> None:
    """Validate all qubits tied in ``meas_map`` are to be acquired.

    Args:
        instruction_map: A dictionary grouping AcquireInstructions according to their start time
                         and the command features (notably, their duration).
        meas_map: List of groups of qubits that must be acquired together.

    Raises:
        QiskitError: If the instructions do not satisfy the measurement map.
    """
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


def _bundle_channel_indices(
        instructions: List[AcquireInstruction]
) -> Tuple[List[int], List[int], List[int]]:
    """From the list of AcquireInstructions, bundle the indices of the acquire channels,
    memory slots, and register slots into a 3-tuple of lists.

    Args:
        instructions: A list of AcquireInstructions to be bundled.

    Returns:
        The qubit indices, the memory slot indices, and register slot indices from instructions.
    """
    qubits = []
    mem_slots = []
    reg_slots = []
    for inst in instructions:
        qubits.extend(aq.index for aq in inst.acquires)
        mem_slots.extend(mem_slot.index for mem_slot in inst.mem_slots)
        reg_slots.extend(reg.index for reg in inst.reg_slots)
    return qubits, mem_slots, reg_slots


def _assemble_config(lo_converter: LoConfigConverter,
                     experiment_config: Dict[str, Any],
                     run_config: RunConfig) -> PulseQobjConfig:
    """Assembles the QobjConfiguration from experimental config and runtime config.

    Args:
        lo_converter: The configured frequency converter and validator.
        experiment_config: Schedules to assemble.
        run_config: Configuration of the runtime environment.

    Returns:
        The assembled PulseQobjConfig.
    """
    qobj_config = run_config.to_dict()
    qobj_config.update(experiment_config)

    # Run config not needed in qobj config
    qobj_config.pop('meas_map', None)
    qobj_config.pop('qubit_lo_range', None)
    qobj_config.pop('meas_lo_range', None)

    # convert enums to serialized values
    meas_return = qobj_config.get('meas_return', 'avg')
    if isinstance(meas_return, MeasReturnType):
        qobj_config['meas_return'] = meas_return.value

    meas_level = qobj_config.get('meas_level', 2)
    if isinstance(meas_level, MeasLevel):
        qobj_config['meas_level'] = meas_level.value

    # convert lo frequencies to Hz
    qobj_config['qubit_lo_freq'] = [freq / 1e9 for freq in qobj_config['qubit_lo_freq']]
    qobj_config['meas_lo_freq'] = [freq / 1e9 for freq in qobj_config['meas_lo_freq']]

    # frequency sweep config
    schedule_los = qobj_config.pop('schedule_los', [])
    if len(schedule_los) == 1:
        lo_dict = schedule_los[0]
        q_los = lo_converter.get_qubit_los(lo_dict)
        # Hz -> GHz
        if q_los:
            qobj_config['qubit_lo_freq'] = [freq / 1e9 for freq in q_los]
        m_los = lo_converter.get_meas_los(lo_dict)
        if m_los:
            qobj_config['meas_lo_freq'] = [freq / 1e9 for freq in m_los]

    return PulseQobjConfig(**qobj_config)
