# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=unused-import

"""Assemble function for converting a list of circuits into a qobj"""
import warnings
import uuid
import logging
import sympy

from qiskit.exceptions import QiskitError
from qiskit.pulse import Schedule, LoConfig
from qiskit.pulse.commands import DriveInstruction
from qiskit.compiler.run_config import RunConfig
from qiskit.qobj import (QasmQobj, PulseQobj, QobjExperimentHeader, QobjHeader,
                         QasmQobjInstruction, QasmQobjExperimentConfig, QasmQobjExperiment,
                         QasmQobjConfig, PulseQobjInstruction, PulseQobjExperimentConfig,
                         PulseQobjExperiment, PulseQobjConfig, QobjPulseLibrary)
from qiskit.qobj.converters import PulseQobjConverter, LoConfigConverter

logger = logging.getLogger(__name__)


def assemble_circuits(circuits, qobj_id=None, qobj_header=None,
                      shots=None, memory=None, max_credits=None,
                      seed_simulator=None, run_config=None,
                      config=None, seed=None):  # deprecated
    """Assembles a list of circuits into a qobj which can be run on the backend.

    This function serializes the circuits to create Qobj "experiments", and
    annotates the experiment payload with header and configurations.

    Args:
        circuits (list[QuantumCircuits] or QuantumCircuit):
            circuit(s) to assemble

        qobj_id (str):
            String identifier to annotate the Qobj

        qobj_header (QobjHeader or dict):
            User input that will be inserted in Qobj header, and will also be
            copied to the corresponding Result header. Headers do not affect the run.

        shots (int):
            number of repetitions of each circuit, for sampling. Default: 2014

        memory (bool):
            if True, per-shot measurement bitstrings are returned as well
            (provided the backend supports it). Default: False

        max_credits (int):
            maximum credits to spend on job. Default: 10

        seed_simulator (int):
            random seed to control sampling, for when backend is a simulator

        config (dict):
            DEPRECATED in 0.8: use run_config instead

        run_config (RunConfig):
            Qobj runtime configuration, containing some or all of the above options.
            If any other option is explicitly set (e.g. shots), it
            will override the run_config's.

        seed (int):
            DEPRECATED in 0.8: use ``seed_simulator`` kwarg instead

    Returns:
        QasmQobj: the Qobj to be run on the backends
    """
    # deprecation matter
    if config:
        warnings.warn('config is not used anymore. Set all configs in '
                      'run_config.', DeprecationWarning)
        run_config = run_config or config
    if seed:
        warnings.warn('seed is deprecated in favor of seed_simulator.', DeprecationWarning)
        seed_simulator = seed_simulator or seed

    # The header that goes at the top of the Qobj (and later Result)
    qobj_header = qobj_header or QobjHeader()
    if isinstance(qobj_header, dict):
        qobj_header = QobjHeader(**qobj_header)

    # Get RunConfig(s) to configure each QASM experiment
    run_config = run_config or RunConfig()
    shots = shots or getattr(run_config, 'shots', None)
    memory = memory or getattr(run_config, 'memory', None)
    max_credits = max_credits or getattr(run_config, 'max_credits', None)
    seed_simulator = (seed_simulator or
                      getattr(run_config, 'seed_simulator', None) or
                      getattr(run_config, 'seed', None))  # deprecated
    # this ensures that None values are not set, otherwise
    # validation fails when 'null' is not acceptable
    run_config_dict = dict(shots=shots,
                           memory=memory,
                           max_credits=max_credits,
                           seed_simulator=seed_simulator,
                           seed=seed_simulator)  # deprecated
    run_config = RunConfig(**{k: v for k, v in run_config_dict.items() if v is not None})

    # Pack everything into the Qobj
    circuits = circuits if isinstance(circuits, list) else [circuits]

    userconfig = QasmQobjConfig(**run_config.to_dict())
    experiments = []
    max_n_qubits = 0
    max_memory_slots = 0
    for circuit in circuits:
        # header stuff
        n_qubits = 0
        memory_slots = 0
        qubit_labels = []
        clbit_labels = []

        qreg_sizes = []
        creg_sizes = []
        for qreg in circuit.qregs:
            qreg_sizes.append([qreg.name, qreg.size])
            for j in range(qreg.size):
                qubit_labels.append([qreg.name, j])
            n_qubits += qreg.size
        for creg in circuit.cregs:
            creg_sizes.append([creg.name, creg.size])
            for j in range(creg.size):
                clbit_labels.append([creg.name, j])
            memory_slots += creg.size

        # TODO: why do we need creq_sizes and qreg_sizes in header
        # TODO: we need to rethink memory_slots as they are tied to classical bit
        experimentheader = QobjExperimentHeader(qubit_labels=qubit_labels,
                                                n_qubits=n_qubits,
                                                qreg_sizes=qreg_sizes,
                                                clbit_labels=clbit_labels,
                                                memory_slots=memory_slots,
                                                creg_sizes=creg_sizes,
                                                name=circuit.name)
        # TODO: why do we need n_qubits and memory_slots in both the header and the config
        experimentconfig = QasmQobjExperimentConfig(n_qubits=n_qubits, memory_slots=memory_slots)

        # Convert conditionals from QASM-style (creg ?= int) to qobj-style
        # (register_bit ?= 1), by assuming device has unlimited register slots
        # (supported only for simulators). Map all measures to a register matching
        # their clbit_index, create a new register slot for every conditional gate
        # and add a bfunc to map the creg=val mask onto the gating register bit.

        is_conditional_experiment = any(op.control for (op, qargs, cargs) in circuit.data)
        max_conditional_idx = 0

        instructions = []
        for op_context in circuit.data:
            op = op_context[0]
            qargs = op_context[1]
            cargs = op_context[2]
            current_instruction = QasmQobjInstruction(name=op.name)
            if qargs:
                qubit_indices = [qubit_labels.index([qubit[0].name, qubit[1]])
                                 for qubit in qargs]
                current_instruction.qubits = qubit_indices
            if cargs:
                clbit_indices = [clbit_labels.index([clbit[0].name, clbit[1]])
                                 for clbit in cargs]
                current_instruction.memory = clbit_indices

                # If the experiment has conditional instructions, assume every
                # measurement result may be needed for a conditional gate.
                if op.name == "measure" and is_conditional_experiment:
                    current_instruction.register = clbit_indices

            if op.params:
                # Evalute Sympy parameters
                params = [
                    x.evalf() if hasattr(x, 'evalf') else x for x in op.params
                ]
                params = [sympy.matrix2numpy(x, dtype=complex)
                          if isinstance(x, sympy.Matrix) else x for x in params]

                current_instruction.params = params
            # TODO: I really dont like this for snapshot. I also think we should change
            # type to snap_type
            if op.name == 'snapshot':
                current_instruction.label = str(op.params[0])
                current_instruction.snapshot_type = str(op.params[1])
            if op.name == 'unitary':
                if op._label:
                    current_instruction.label = op._label
            if op.control:
                # To convert to a qobj-style conditional, insert a bfunc prior
                # to the conditional instruction to map the creg ?= val condition
                # onto a gating register bit.
                mask = 0
                val = 0

                for clbit in clbit_labels:
                    if clbit[0] == op.control[0].name:
                        mask |= (1 << clbit_labels.index(clbit))
                        val |= (((op.control[1] >> clbit[1]) & 1) << clbit_labels.index(clbit))

                conditional_reg_idx = memory_slots + max_conditional_idx
                conversion_bfunc = QasmQobjInstruction(name='bfunc',
                                                       mask="0x%X" % mask,
                                                       relation='==',
                                                       val="0x%X" % val,
                                                       register=conditional_reg_idx)
                instructions.append(conversion_bfunc)

                current_instruction.conditional = conditional_reg_idx
                max_conditional_idx += 1

            instructions.append(current_instruction)

        experiments.append(QasmQobjExperiment(instructions=instructions, header=experimentheader,
                                              config=experimentconfig))
        if n_qubits > max_n_qubits:
            max_n_qubits = n_qubits
        if memory_slots > max_memory_slots:
            max_memory_slots = memory_slots

    userconfig.memory_slots = max_memory_slots
    userconfig.n_qubits = max_n_qubits

    return QasmQobj(qobj_id=qobj_id or str(uuid.uuid4()),
                    config=userconfig,
                    experiments=experiments,
                    header=qobj_header)


def assemble_schedules(schedules, default_qubit_los, default_meas_los,
                       schedule_los=None, shots=1024, qobj_id=None,
                       meas_level=2, meas_return='avg', memory=None,
                       memory_slots=None, memory_slot_size=100,
                       rep_time=None, max_credits=10, seed=None,
                       qobj_header=None, instruction_converter=PulseQobjConverter,
                       **run_config):
    """Assembles a list of circuits into a qobj which can be run on the backend.

    Args:
        schedules (list[Schedule] or Schedule): schedules to assemble
        default_qubit_los (list): List of default qubit lo frequencies
        default_meas_los (list): List of default meas lo frequencies
        qobj_header (QobjHeader or dict): header to pass to the results
        schedule_los(None or list[Union[Dict[OutputChannel, float], LoConfig]] or
                        Union[Dict[OutputChannel, float], LoConfig]): Experiment LO configurations
        shots (int): number of repetitions of each circuit, for sampling
        qobj_id (int): identifier for the generated qobj
        meas_level (int): set the appropriate level of the measurement output.
        meas_return (str): indicates the level of measurement data for the backend to return
            for `meas_level` 0 and 1:
                "single" returns information from every shot of the experiment.
                "avg" returns the average measurement output (averaged over the number of shots).
        memory (bool or None): For `meas_level` 2, return the individual shot results.
        memory_slots (int): number of classical memory slots used in this job.
        memory_slot_size (int): size of each memory slot if the output is Level 0.
        rep_time (int): repetition time of the experiment in μs.
            The delay between experiments will be rep_time.
            Must be from the list provided by the device.
        max_credits (int): maximum credits to use
        seed (int): random seed for simulators
        instruction_converter (PulseQobjConverter): converter for pulse instruction
        run_config: Additional keyword arguments to be inserted in the Qobj configuration.

    Returns:
        PulseQobj: the Qobj to be run on the backends

    Raises:
        QiskitError: when invalid schedules or configs are provided
    """
    if isinstance(schedules, Schedule):
        schedules = [schedules]

    # add default empty lo config
    if schedule_los is None:
        schedule_los = []

    if isinstance(schedule_los, (LoConfig, dict)):
        schedule_los = [schedule_los]

    if qobj_id is None:
        qobj_id = str(uuid.uuid4())

    # Convert to LoConfig if lo configuration supplied as dictionary
    schedule_los = [lo_config if isinstance(lo_config, LoConfig) else LoConfig(lo_config)
                    for lo_config in schedule_los]

    user_pulselib = set()

    qobj_header = qobj_header or QobjHeader()
    if isinstance(qobj_header, dict):
        qobj_header = QobjHeader(**qobj_header)
    # create run configuration and populate
    run_config['qubit_lo_freq'] = default_qubit_los
    run_config['meas_lo_freq'] = default_meas_los
    run_config['shots'] = shots
    run_config['max_credits'] = max_credits
    run_config['meas_level'] = meas_level

    if meas_level == 2:
        if meas_return == 'avg':
            logger.warning('"meas_return=avg" is not a supported option for "meas_level=2".'
                           'If you wish to obtain the binned counts please use "meas_level=2" and'
                           'if you wish to obtain the individual shot results, set "memory=True".')
        memory = True

        run_config['memory'] = memory
        run_config['meas_return'] = 'single'
    else:
        run_config['meas_return'] = meas_return
        if memory:
            logger.warning('Setting "memory" does not have an effect for "meas_level=0/1".')

    run_config['memory_slot'] = memory_slots
    run_config['memory_slot_size'] = memory_slot_size
    run_config['rep_time'] = rep_time
    if seed:
        run_config['seed'] = seed

    instruction_converter = instruction_converter(PulseQobjInstruction, **run_config)
    lo_converter = LoConfigConverter(PulseQobjExperimentConfig, default_qubit_los,
                                     default_meas_los, **run_config)

    # assemble schedules
    qobj_schedules = []
    for idx, schedule in enumerate(schedules):
        # instructions
        qobj_instructions = []
        for instruction in schedule.flat_instruction_sequence():
            # TODO: support conditional gate
            qobj_instructions.append(instruction_converter(instruction))
            if isinstance(instruction, DriveInstruction):
                # add samples to pulse library
                user_pulselib.add(instruction.command)
        # experiment header
        qobj_experiment_header = QobjExperimentHeader(
            name=schedule.name or 'Experiment-%d' % idx
        )

        qobj_schedules.append({
            'header': qobj_experiment_header,
            'instructions': qobj_instructions
        })

    # setup pulse_library
    run_config['pulse_library'] = [QobjPulseLibrary(name=pulse.name, samples=pulse.samples)
                                   for pulse in user_pulselib]

    # create qob experiment field
    experiments = []
    if len(schedule_los) == 1:
        lo_dict = schedule_los.pop()
        # update global config
        q_los = lo_converter.get_qubit_los(lo_dict)
        if q_los:
            run_config['qubit_lo_freq'] = q_los
        m_los = lo_converter.get_meas_los(lo_dict)
        if m_los:
            run_config['meas_lo_freq'] = m_los

    if schedule_los:
        # multiple frequency setups
        if len(qobj_schedules) == 1:
            # frequency sweep
            for lo_dict in schedule_los:
                experiments.append(PulseQobjExperiment(
                    instructions=qobj_schedules[0]['instructions'],
                    experimentheader=qobj_schedules[0]['header'],
                    experimentconfig=lo_converter(lo_dict)
                ))
        elif len(qobj_schedules) == len(schedule_los):
            # n:n setup
            for lo_dict, schedule in zip(schedule_los, qobj_schedules):
                experiments.append(PulseQobjExperiment(
                    instructions=schedule['instructions'],
                    experimentheader=schedule['header'],
                    experimentconfig=lo_converter(lo_dict)
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
                experimentheader=schedule['header'],
            ))

    qobj_config = PulseQobjConfig(**run_config)

    return PulseQobj(qobj_id=qobj_id,
                     config=qobj_config,
                     experiments=experiments,
                     header=qobj_header)
