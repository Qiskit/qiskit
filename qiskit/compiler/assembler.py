# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Assemble function for converting a list of circuits into a qobj"""
import uuid

import numpy
import sympy

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.pulse import Schedule, Snapshot
from qiskit.pulse.commands import (DriveInstruction, FrameChangeInstruction,
                                   PersistentValueInstruction, AcquireInstruction)
from qiskit.compiler.run_config import RunConfig
from qiskit.qobj import (QasmQobj, PulseQobj, QobjExperimentHeader, QobjHeader,
                         QasmQobjInstruction, QasmQobjExperimentConfig, QasmQobjExperiment,
                         QasmQobjConfig, QobjConditional,
                         PulseQobjInstruction, PulseQobjExperimentConfig, PulseQobjExperiment,
                         PulseQobjConfig, QobjPulseLibrary, QobjMeasurementOption)
from qiskit.exceptions import QiskitError


def assemble_circuits(circuits, run_config=None, qobj_header=None, qobj_id=None):
    """Assembles a list of circuits into a qobj which can be run on the backend.

    Args:
        circuits (list[QuantumCircuits] or QuantumCircuit): circuits to assemble
        run_config (RunConfig): RunConfig object
        qobj_header (QobjHeader): header to pass to the results
        qobj_id (int): identifier for the generated qobj

    Returns:
        QasmQobj: the Qobj to be run on the backends
    """
    qobj_header = qobj_header or QobjHeader()
    run_config = run_config or RunConfig()
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]

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

        instructions = []
        for opt in circuit.data:
            current_instruction = QasmQobjInstruction(name=opt.name)
            if opt.qargs:
                qubit_indices = [qubit_labels.index([qubit[0].name, qubit[1]])
                                 for qubit in opt.qargs]
                current_instruction.qubits = qubit_indices
            if opt.cargs:
                clbit_indices = [clbit_labels.index([clbit[0].name, clbit[1]])
                                 for clbit in opt.cargs]
                current_instruction.memory = clbit_indices

            if opt.params:
                params = list(map(lambda x: x.evalf(), opt.params))
                params = [sympy.matrix2numpy(x, dtype=complex)
                          if isinstance(x, sympy.Matrix) else x for x in params]
                if len(params) == 1 and isinstance(params[0], numpy.ndarray):
                    # TODO: Aer expects list of rows for unitary instruction params;
                    # change to matrix in Aer.
                    params = params[0]
                current_instruction.params = params
            # TODO (jay): I really dont like this for snapshot. I also think we should change
            # type to snap_type
            if opt.name == "snapshot":
                current_instruction.label = str(opt.params[0])
                current_instruction.type = str(opt.params[1])
            if opt.control:
                mask = 0
                for clbit in clbit_labels:
                    if clbit[0] == opt.control[0].name:
                        mask |= (1 << clbit_labels.index(clbit))

                current_instruction.conditional = QobjConditional(mask="0x%X" % mask,
                                                                  type='equals',
                                                                  val="0x%X" % opt.control[1])

            instructions.append(current_instruction)
        experiments.append(QasmQobjExperiment(instructions=instructions, header=experimentheader,
                                              config=experimentconfig))
        if n_qubits > max_n_qubits:
            max_n_qubits = n_qubits
        if memory_slots > max_memory_slots:
            max_memory_slots = memory_slots

    userconfig.memory_slots = max_memory_slots
    userconfig.n_qubits = max_n_qubits

    return QasmQobj(qobj_id=qobj_id or str(uuid.uuid4()), config=userconfig,
                    experiments=experiments, header=qobj_header)


def assemble_schedules(schedules, dict_config, dict_header):
    """Assembles a list of circuits into a qobj which can be run on the backend.

    Args:
        schedules (list[Schedule] or PulseSchedule): schedules to assemble
        dict_config (dict): configuration of experiments
        dict_header (dict): header to pass to the results

    Returns:
        PulseQobj: the Qobj to be run on the backends

    Raises:
        QiskitError: when invalid command is provided
    """
    if isinstance(schedules, Schedule):
        schedules = [schedules]

    experiments = []

    for ii, schedule in enumerate(schedules):

        # use LO frequency configs
        default_qubit_lo_freq = dict_config.get('qubit_lo_freq', None)
        default_meas_lo_freq = dict_config.get('meas_lo_freq', None)

        # TODO: scheudle has no information of LO frequency
        # lo_freqs = {}
        # user_qubit_lo_freq = [q.drive.lo_frequency for q in schedule.device.q]
        # if default_qubit_lo_freq:
        #     if default_qubit_lo_freq != user_qubit_lo_freq:
        #         lo_freqs['qubit_lo_freq'] = user_qubit_lo_freq
        # else:
        #     lo_freqs['qubit_lo_freq'] = user_qubit_lo_freq
        # user_meas_lo_freq = [q.measure.lo_frequency for q in schedule.device.q]
        # if default_meas_lo_freq:
        #     if default_meas_lo_freq != user_meas_lo_freq:
        #         lo_freqs['meas_lo_freq'] = user_meas_lo_freq
        # else:
        #     lo_freqs['meas_lo_freq'] = user_meas_lo_freq

        # generate experimental configuration
        experimentconfig = PulseQobjExperimentConfig(**lo_freqs)

        # generate experimental header
        experimentheader = QobjExperimentHeader(name=schedule.name or 'Schedule-%d' % ii)

        commands = []
        user_pulselib = []
        for block in schedule.flat_instruction_sequence():
            pulse_instr = block.instruction
            current_command = PulseQobjInstruction(name=pulse_instr.name,
                                                   t0=block.begin_time)
            if isinstance(pulse_instr, DriveInstruction):
                # Sample pulses
                # required: `ch`
                # optional:
                current_command.ch = pulse_instr.channel.name
                # TODO: support conditional gate
                if pulse_instr.command not in [p for p in user_pulselib]:
                    user_pulselib.append(pulse_instr.command)
            elif isinstance(pulse_instr, FrameChangeInstruction):
                # Frame change
                # required: `ch`, `phase`
                # optional:
                current_command.ch = pulse_instr.channel.name
                current_command.phase = pulse_instr.command.phase
            elif isinstance(pulse_instr, PersistentValueInstruction):
                # Persistent value
                # required: `ch`, `val`
                # optional:
                current_command.ch = pulse_instr.channel.name
                current_command.val = pulse_instr.command.value
            elif isinstance(pulse_instr, AcquireInstruction):
                # Acquire
                # required: `duration`, `qubits`, `memory_slot`
                # optional: `discriminators`, `kernels`, `register_slot`
                current_command.duration = pulse_instr.duration
                current_command.qubits = [acqs.index for acqs in pulse_instr.acquire_channels]
                current_command.memory_slot = [mems.index for mems in pulse_instr.mem_slots]
                if dict_config.get('meas_level', 2) == 2:
                    # apply discriminator for level 2 measurement
                    current_command.register_slot = [regs.index for regs in pulse_instr.reg_slots]
                    _discriminator = pulse_instr.command.discriminator
                    if _discriminator:
                        qobj_discriminator = QobjMeasurementOption(name=_discriminator.name,
                                                                   params=_discriminator.params)
                        current_command.discriminators = [qobj_discriminator]
                    else:
                        current_command.discriminators = []
                if dict_config.get('meas_level', 2) >= 1:
                    # apply kernel for level 1, 2 measurements
                    _kernel = pulse_instr.command.kernel
                    if _kernel:
                        qobj_kernel = QobjMeasurementOption(name=_kernel.name,
                                                            params=_kernel.params)
                        current_command.kernels = [qobj_kernel]
                    else:
                        current_command.kernels = []
            elif isinstance(pulse_instr, Snapshot):
                # Snapshot
                # required: `label`, `type`
                # optional:
                current_command.label = pulse_instr.label
                current_command.type = pulse_instr.type
            else:
                raise QiskitError('Invalid command is given, %s' % pulse_instr.command.name)

            commands.append(current_command)

        experiments.append(PulseQobjExperiment(instructions=commands,
                                               header=experimentheader,
                                               config=experimentconfig))

    # generate qobj pulse library
    qobj_default_pulselib = list(map(lambda p:
                                     QobjPulseLibrary(name=p['name'], samples=p['samples']),
                                     dict_config.get('pulse_library', []))
                                 )
    qobj_user_pulselib = list(map(lambda p:
                                  QobjPulseLibrary(name=p.name, samples=p.samples),
                                  user_pulselib)
                              )

    dict_config['pulse_library'] = qobj_default_pulselib + qobj_user_pulselib

    qobj_config = PulseQobjConfig(**dict_config)
    qobj_header = QobjHeader(**dict_header)

    return PulseQobj(qobj_id=str(uuid.uuid4()), config=qobj_config,
                     experiments=experiments, header=qobj_header)
