# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Assemble function for converting a list of circuits into a qobj"""
import warnings
import uuid
import logging
import sympy

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.pulse import Schedule, LoConfig
from qiskit.pulse.commands import DriveInstruction
from qiskit.compiler.run_config import RunConfig
from qiskit.qobj import (QasmQobj, PulseQobj, QobjExperimentHeader, QobjHeader,
                         QasmQobjInstruction, QasmQobjExperimentConfig, QasmQobjExperiment,
                         QasmQobjConfig, PulseQobjInstruction, PulseQobjExperimentConfig,
                         PulseQobjExperiment, PulseQobjConfig, QobjPulseLibrary)
from qiskit.qobj.converters import PulseQobjConverter, LoConfigConverter
from qiskit.validation.exceptions import ModelValidationError

logger = logging.getLogger(__name__)


def assemble_circuits(circuits, qobj_id=None, qobj_header=None, run_config=None):
    """Assembles a list of circuits into a qobj which can be run on the backend.

    Args:
        circuits (list[QuantumCircuits]): circuit(s) to assemble
        qobj_id (int): identifier for the generated qobj
        qobj_header (QobjHeader): header to pass to the results
        run_config (RunConfig): configuration of the runtime environment

    Returns:
        QasmQobj: the Qobj to be run on the backends
    """
    qobj_config = QasmQobjConfig()
    if run_config:
        qobj_config = QasmQobjConfig(**run_config.to_dict())

    # Pack everything into the Qobj
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

    qobj_config.memory_slots = max_memory_slots
    qobj_config.n_qubits = max_n_qubits

    return QasmQobj(qobj_id=qobj_id,
                    config=qobj_config,
                    experiments=experiments,
                    header=qobj_header)


def assemble_schedules(schedules, qobj_id=None, qobj_header=None, run_config=None):
    """Assembles a list of schedules into a qobj which can be run on the backend.

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
    qobj_config = QasmQobjConfig()
    if run_config:
        qobj_config = QasmQobjConfig(**run_config.to_dict())

    # Get appropriate convertors
    instruction_converter = PulseQobjConverter
    instruction_converter = instruction_converter(PulseQobjInstruction, **run_config.to_dict())
    lo_converter = LoConfigConverter(PulseQobjExperimentConfig, run_config.qubit_lo_freq,
                                     run_config.meas_lo_freq, **run_config.to_dict())

    # Pack everything into the Qobj
    qobj_schedules = []
    user_pulselib = set()
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
    run_config.pulse_library = [QobjPulseLibrary(name=pulse.name, samples=pulse.samples)
                                for pulse in user_pulselib]

    # create qob experiment field
    experiments = []
    if len(run_config.schedule_los) == 1:
        lo_dict = run_config.schedule_los.pop()
        # update global config
        q_los = lo_converter.get_qubit_los(lo_dict)
        if q_los:
            run_config.qubit_lo_freq = q_los
        m_los = lo_converter.get_meas_los(lo_dict)
        if m_los:
            run_config.meas_lo_freq = m_los

    if run_config.schedule_los:
        # multiple frequency setups
        if len(qobj_schedules) == 1:
            # frequency sweep
            for lo_dict in run_config.schedule_los:
                experiments.append(PulseQobjExperiment(
                    instructions=qobj_schedules[0]['instructions'],
                    experimentheader=qobj_schedules[0]['header'],
                    experimentconfig=lo_converter(lo_dict)
                ))
        elif len(qobj_schedules) == len(run_config.schedule_los):
            # n:n setup
            for lo_dict, schedule in zip(run_config.schedule_los, qobj_schedules):
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

    qobj_config = PulseQobjConfig(**run_config.to_dict())

    return PulseQobj(qobj_id=qobj_id,
                     config=qobj_config,
                     experiments=experiments,
                     header=qobj_header)


# TODO: parallelize over the experiments (serialize each separately, then add global header/config)
def assemble(experiments,
             qobj_id=None, qobj_header=None,  # common run options
             shots=None, memory=None, max_credits=None,
             seed_simulator=None,
             default_qubit_los=None, default_meas_los=None,  # schedule run options
             schedule_los=None, meas_level=2, meas_return='avg',
             memory_slots=None, memory_slot_size=100, rep_time=None,
             config=None, seed=None,  # deprecated
             backend=None, **run_config):
    """Assemble a list of circuits or pulse schedules into a Qobj.

    This function serializes the payloads, which could be either circuits or schedules,
    to create Qobj "experiments". It further annotates the experiment payload with
    header and configurations.

    Args:
        experiments (QuantumCircuit or list[QuantumCircuit] or Schedule or list[Schedule]):
            Circuit(s) or pulse schedule(s) to execute

        qobj_id (str):
            String identifier to annotate the Qobj

        qobj_header (QobjHeader or dict):
            User input that will be inserted in Qobj header, and will also be
            copied to the corresponding Result header. Headers do not affect the run.

        shots (int):
            Number of repetitions of each circuit, for sampling. Default: 2014

        memory (bool):
            If True, per-shot measurement bitstrings are returned as well
            (provided the backend supports it). For OpenPulse jobs, only
            measurement level 2 supports this option. Default: False

        max_credits (int):
            Maximum credits to spend on job. Default: 10

        seed_simulator (int):
            Random seed to control sampling, for when backend is a simulator

        default_qubit_los (list):
            List of default qubit lo frequencies

        default_meas_los (list):
            List of default meas lo frequencies

        schedule_los (None or list[Union[Dict[OutputChannel, float], LoConfig]] or
                      Union[Dict[OutputChannel, float], LoConfig]):
            Experiment LO configurations

        meas_level (int):
            Set the appropriate level of the measurement output for pulse experiments.

        meas_return (str):
            Level of measurement data for the backend to return
            For `meas_level` 0 and 1:
                "single" returns information from every shot.
                "avg" returns average measurement output (averaged over number of shots).

        memory_slots (int):
            Number of classical memory slots used in this job.

        memory_slot_size (int):
            Size of each memory slot if the output is Level 0.

        rep_time (int): repetition time of the experiment in μs.
            The delay between experiments will be rep_time.
            Must be from the list provided by the device.

        backend (BaseBackend):
            If set, some runtime options are automatically grabbed from
            backend.configuration() and backend.defaults().
            If any other option is explicitly set (e.g. rep_rate), it
            will override the backend's.
            If any other options is set in the run_config, it will
            also override the backend's.

        seed (int):
            DEPRECATED in 0.8: use ``seed_simulator`` kwarg instead

        config (dict):
            DEPRECATED in 0.8: use run_config instead

        run_config (dict):
            extra arguments used to configure the run (e.g. for Aer configurable backends)
            Refer to the backend documentation for details on these arguments

    Returns:
        Qobj: a qobj which can be run on a backend. Depending on the type of input,
            this will be either a QasmQobj or a PulseQobj.

    Raises:
        QiskitError: if the input cannot be interpreted as either circuits or schedules
    """
    # deprecation matter
    if config:
        warnings.warn('config is not used anymore. Set all configs in '
                      'run_config.', DeprecationWarning)
        run_config = run_config or config
    if seed:
        warnings.warn('seed is deprecated in favor of seed_simulator.', DeprecationWarning)
        seed_simulator = seed_simulator or seed

    # Get RunConfig(s) that will be inserted in Qobj to configure the run
    experiments = experiments if isinstance(experiments, list) else [experiments]
    qobj_id, qobj_header, run_config = _parse_run_args(qobj_id, qobj_header,
                                                       shots, memory, max_credits, seed_simulator,
                                                       default_qubit_los, default_meas_los,
                                                       schedule_los, meas_level, meas_return,
                                                       memory_slots, memory_slot_size, rep_time,
                                                       backend, **run_config)

    # assemble either circuits or schedules
    if all(isinstance(exp, QuantumCircuit) for exp in experiments):
        return assemble_circuits(circuits=experiments, qobj_id=qobj_id,
                                 qobj_header=qobj_header, run_config=run_config)

    elif all(isinstance(exp, Schedule) for exp in experiments):
        return assemble_schedules(schedules=experiments, qobj_id=qobj_id,
                                  qobj_header=qobj_header, run_config=run_config)

    else:
        raise QiskitError("bad input to assemble() function; "
                          "must be either circuits or schedules")


# TODO: rework to return a list of RunConfigs (one for each experiments), and a global one
def _parse_run_args(qobj_id, qobj_header,
                    shots, memory, max_credits, seed_simulator,
                    default_qubit_los, default_meas_los,
                    schedule_los, meas_level, meas_return,
                    memory_slots, memory_slot_size, rep_time,
                    backend, **run_config):
    """Resolve the various types of args allowed to the assemble() function through
    duck typing, overriding args, etc. Refer to the assemble() docstring for details on
    what types of inputs are allowed.

    Here the args are resolved by converting them to standard instances, and prioritizing
    them in case a run option is passed through multiple args (explicitly setting an arg
    has more priority than the arg set by backend)

    Returns:
        RunConfig: a run config, which is a standardized object that configures the qobj
            and determines the runtime environment.
    """
    # grab relevant info from backend if it exists
    backend_config = None
    backend_default = None
    if backend:
        backend_config = backend.configuration()
        # TODO : Remove usage of config.defaults when backend.defaults() is updated.
        try:
            backend_default = backend.defaults()
        except (ModelValidationError, AttributeError):
            from collections import namedtuple
            backend_config_defaults = getattr(backend_config, 'defaults', {})
            BackendDefault = namedtuple('BackendDefault', ('qubit_freq_est', 'meas_freq_est'))
            backend_default = BackendDefault(
                qubit_freq_est=backend_config_defaults.get('qubit_freq_est'),
                meas_freq_est=backend_config_defaults.get('meas_freq_est')
            )

    memory_slots = memory_slots or getattr(backend_config, 'memory_slots', None)
    rep_time = rep_time or getattr(backend_config, 'rep_times', None)
    if isinstance(rep_time, list):
        rep_time = rep_time[-1]

    # add default empty lo config
    schedule_los = schedule_los or []
    if isinstance(schedule_los, (LoConfig, dict)):
        schedule_los = [schedule_los]

    # Convert to LoConfig if lo configuration supplied as dictionary
    schedule_los = [lo_config if isinstance(lo_config, LoConfig) else LoConfig(lo_config)
                    for lo_config in schedule_los]

    qubit_lo_freq = default_qubit_los or getattr(backend_default, 'qubit_freq_est', [])
    meas_lo_freq = default_meas_los or getattr(backend_default, 'meas_freq_est', [])

    if meas_level == 2:
        if meas_return == 'avg':
            logger.warning('"meas_return=avg" is not a supported option for "meas_level=2".'
                           'If you wish to obtain the binned counts please use "meas_level=2" and'
                           'if you wish to obtain the individual shot results, set "memory=True".')
        memory = True
        meas_return = 'single'
    else:
        if memory:
            logger.warning('Setting "memory" does not have an effect for "meas_level=0/1".')

    # an identifier for the Qobj
    qobj_id = qobj_id or str(uuid.uuid4())

    # The header that goes at the top of the Qobj (and later Result)
    # we process it as dict, then write entries that are not None to a QobjHeader object
    qobj_header = qobj_header or {}
    if isinstance(qobj_header, QobjHeader):
        qobj_header = qobj_header.to_dict()
    backend_name = getattr(backend_config, 'backend_name', None)
    backend_version = getattr(backend_config, 'backend_version', None)
    qobj_header = {**dict(backend_name=backend_name, backend_version=backend_version),
                   **qobj_header}
    qobj_header = QobjHeader(**{k: v for k, v in qobj_header.items() if v is not None})

    # create run configuration and populate
    run_config_dict = dict(shots=shots,
                           memory=memory,
                           max_credits=max_credits,
                           seed_simulator=seed_simulator,
                           seed=seed_simulator,  # deprecated
                           qubit_lo_freq=qubit_lo_freq,
                           meas_lo_freq=meas_lo_freq,
                           schedule_los=schedule_los,
                           meas_level=meas_level,
                           meas_return=meas_return,
                           memory_slots=memory_slots,
                           memory_slot_size=memory_slot_size,
                           rep_time=rep_time,
                           **run_config)
    run_config = RunConfig(**{k: v for k, v in run_config_dict.items() if v is not None})

    return qobj_id, qobj_header, run_config
