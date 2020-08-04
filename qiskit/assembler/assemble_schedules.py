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
from typing import Any, Dict, List, Tuple
import hashlib

from qiskit import qobj, pulse
from qiskit.assembler.run_config import RunConfig
from qiskit.exceptions import QiskitError
from qiskit.pulse import instructions, transforms, library
from qiskit.qobj import utils as qobj_utils, converters
from qiskit.qobj.converters.pulse_instruction import ParametricPulseShapes


def assemble_schedules(schedules: List[pulse.Schedule],
                       qobj_id: int,
                       qobj_header: qobj.QobjHeader,
                       run_config: RunConfig) -> qobj.PulseQobj:
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

    lo_converter = converters.LoConfigConverter(qobj.PulseQobjExperimentConfig,
                                                **run_config.to_dict())
    experiments, experiment_config = _assemble_experiments(schedules,
                                                           lo_converter,
                                                           run_config)
    qobj_config = _assemble_config(lo_converter, experiment_config, run_config)

    return qobj.PulseQobj(experiments=experiments,
                          qobj_id=qobj_id,
                          header=qobj_header,
                          config=qobj_config)


def _assemble_experiments(
        schedules: List[pulse.Schedule],
        lo_converter: converters.LoConfigConverter,
        run_config: RunConfig
) -> Tuple[List[qobj.PulseQobjExperiment], Dict[str, Any]]:
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

    instruction_converter = getattr(run_config,
                                    'instruction_converter',
                                    converters.InstructionToQobjConverter)
    instruction_converter = instruction_converter(qobj.PulseQobjInstruction,
                                                  **run_config.to_dict())
    compressed_schedules = transforms.compress_pulses(schedules)

    user_pulselib = {}
    experiments = []
    for idx, schedule in enumerate(compressed_schedules):
        qobj_instructions, max_memory_slot = _assemble_instructions(
            schedule,
            instruction_converter,
            run_config,
            user_pulselib)

        # TODO: add other experimental header items (see circuit assembler)
        qobj_experiment_header = qobj.QobjExperimentHeader(
            memory_slots=max_memory_slot + 1,  # Memory slots are 0 indexed
            name=schedule.name or 'Experiment-%d' % idx)

        experiment = qobj.PulseQobjExperiment(
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
            experiments.append(qobj.PulseQobjExperiment(
                header=experiment.header,
                instructions=experiment.instructions,
                config=freq_config))

    # Top level Qobj configuration
    experiment_config = {
        'pulse_library': [qobj.PulseLibraryItem(name=name, samples=samples)
                          for name, samples in user_pulselib.items()],
        'memory_slots': max([exp.header.memory_slots for exp in experiments])
    }

    return experiments, experiment_config


def _assemble_instructions(
        schedule: pulse.Schedule,
        instruction_converter: converters.InstructionToQobjConverter,
        run_config: RunConfig,
        user_pulselib: Dict[str, List[complex]]
) -> Tuple[List[qobj.PulseQobjInstruction], int]:
    """Assembles the instructions in a schedule into a list of PulseQobjInstructions and returns
    related metadata that will be assembled into the Qobj configuration. Lookup table for
    pulses defined in all experiments are registered in ``user_pulselib``. This object should be
    mutable python dictionary so that items are properly updated after each instruction assemble.
    The dictionary is not returned to avoid redundancy.

    Args:
        schedule: Schedule to assemble.
        instruction_converter: A converter instance which can convert PulseInstructions to
                               PulseQobjInstructions.
        run_config: Configuration of the runtime environment.
        user_pulselib: User pulse library from previous schedule.

    Returns:
        A list of converted instructions, the user pulse library dictionary (from pulse name to
        pulse samples), and the maximum number of readout memory slots used by this Schedule.
    """
    max_memory_slot = 0
    qobj_instructions = []

    acquire_instruction_map = defaultdict(list)
    for time, instruction in schedule.instructions:

        if (isinstance(instruction, instructions.Play) and
                isinstance(instruction.pulse, library.ParametricPulse)):
            pulse_shape = ParametricPulseShapes(type(instruction.pulse)).name
            if pulse_shape not in run_config.parametric_pulses:
                instruction = instructions.Play(instruction.pulse.get_sample_pulse(),
                                                instruction.channel,
                                                name=instruction.name)

        if (isinstance(instruction, instructions.Play) and
                isinstance(instruction.pulse, library.Waveform)):
            name = hashlib.sha256(instruction.pulse.samples).hexdigest()
            instruction = instructions.Play(
                library.Waveform(name=name, samples=instruction.pulse.samples),
                channel=instruction.channel,
                name=name)
            user_pulselib[name] = instruction.pulse.samples

        if isinstance(instruction, instructions.Acquire):
            if instruction.mem_slot:
                max_memory_slot = max(max_memory_slot, instruction.mem_slot.index)
            # Acquires have a single AcquireChannel per inst, but we have to bundle them
            # together into the Qobj as one instruction with many channels
            acquire_instruction_map[(time, instruction.duration)].append(instruction)
            continue

        if isinstance(instruction, (instructions.Delay, instructions.Directive)):
            # delay instructions are ignored as timing is explicit within qobj
            continue

        qobj_instructions.append(instruction_converter(time, instruction))

    if acquire_instruction_map:
        if hasattr(run_config, 'meas_map'):
            _validate_meas_map(acquire_instruction_map, run_config.meas_map)
        for (time, _), instrs in acquire_instruction_map.items():
            qobj_instructions.append(
                instruction_converter.convert_bundled_acquires(
                    time,
                    instrs
                ),
            )

    return qobj_instructions, max_memory_slot


def _validate_meas_map(instruction_map: Dict[Tuple[int, instructions.Acquire],
                                             List[instructions.Acquire]],
                       meas_map: List[List[int]]) -> None:
    """Validate all qubits tied in ``meas_map`` are to be acquired.

    Args:
        instruction_map: A dictionary grouping Acquire instructions according to their start time
                         and duration.
        meas_map: List of groups of qubits that must be acquired together.

    Raises:
        QiskitError: If the instructions do not satisfy the measurement map.
    """
    meas_map_sets = [set(m) for m in meas_map]

    # Check each acquisition time individually
    for _, instrs in instruction_map.items():
        measured_qubits = set()
        for inst in instrs:
            measured_qubits.add(inst.channel.index)

        for meas_set in meas_map_sets:
            intersection = measured_qubits.intersection(meas_set)
            if intersection and intersection != meas_set:
                raise QiskitError('Qubits to be acquired: {0} do not satisfy required qubits '
                                  'in measurement map: {1}'.format(measured_qubits, meas_set))


def _assemble_config(lo_converter: converters.LoConfigConverter,
                     experiment_config: Dict[str, Any],
                     run_config: RunConfig) -> qobj.PulseQobjConfig:
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
    if isinstance(meas_return, qobj_utils.MeasReturnType):
        qobj_config['meas_return'] = meas_return.value

    meas_level = qobj_config.get('meas_level', 2)
    if isinstance(meas_level, qobj_utils.MeasLevel):
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

    return qobj.PulseQobjConfig(**qobj_config)
