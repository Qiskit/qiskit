# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=unused-import

"""Assemble function for converting a list of circuits into a qobj"""
import uuid

import numpy
import sympy

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.pulse import Schedule, LoConfig
from qiskit.pulse.commands import DriveInstruction
from qiskit.qobj import (QasmQobj, PulseQobj, QobjExperimentHeader, QobjHeader,
                         QasmQobjInstruction, QasmQobjExperimentConfig, QasmQobjExperiment,
                         QasmQobjConfig,
                         PulseQobjInstruction, PulseQobjExperimentConfig, PulseQobjExperiment,
                         PulseQobjConfig, QobjPulseLibrary)
from qiskit.qobj.converters import PulseQobjConverter, LoDictConverter
from .run_config import RunConfig


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
                params = list(map(lambda x: x.evalf(), op.params))
                params = [sympy.matrix2numpy(x, dtype=complex)
                          if isinstance(x, sympy.Matrix) else x for x in params]
                if len(params) == 1 and isinstance(params[0], numpy.ndarray):
                    # TODO: Aer expects list of rows for unitary instruction params;
                    # change to matrix in Aer.
                    params = params[0]
                current_instruction.params = params
            # TODO: I really dont like this for snapshot. I also think we should change
            # type to snap_type
            if op.name == "snapshot":
                current_instruction.label = str(op.params[0])
                current_instruction.type = str(op.params[1])
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

    return QasmQobj(qobj_id=qobj_id or str(uuid.uuid4()), config=userconfig,
                    experiments=experiments, header=qobj_header)


def assemble_schedules(schedules, user_lo_configs,
                       dict_config, dict_header,
                       inst_converter=PulseQobjConverter):
    """Assembles a list of circuits into a qobj which can be run on the backend.

    Args:
        schedules (list[Schedule] or Schedule): schedules to assemble
        user_lo_configs(list[LoConfig] or LoConfig): LO dictionary to assemble
        dict_config (dict): configuration of experiments
        dict_header (dict): header to pass to the results
        inst_converter (PulseQobjConverter): converter for pulse instruction

    Returns:
        PulseQobj: the Qobj to be run on the backends

    Raises:
        QiskitError: when invalid schedules or configs are provided
    """

    _inst_converter = inst_converter(PulseQobjInstruction, **dict_config)
    _lo_converter = LoDictConverter(PulseQobjExperimentConfig, **dict_config)

    if isinstance(schedules, Schedule):
        schedules = [schedules]

    if isinstance(user_lo_configs, LoConfig):
        user_lo_configs = [user_lo_configs]

    user_pulselib = set()

    # assemble schedules
    qobj_schedules = []
    for idx, schedule in enumerate(schedules):
        # instructions
        qobj_instructions = []
        for instruction in schedule.flat_instruction_sequence():
            # TODO: support conditional gate
            qobj_instructions.append(_inst_converter(instruction))
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
    dict_config['pulse_library'] = [
        QobjPulseLibrary(name=pulse.name, samples=pulse.samples) for pulse in user_pulselib
    ]

    # create qob experiment field
    experiments = []
    if len(user_lo_configs) == 1:
        lo_dict = user_lo_configs.pop()
        # update global config
        q_los = _lo_converter.get_qubit_los(lo_dict)
        if q_los:
            dict_config['qubit_lo_freq'] = q_los
        m_los = _lo_converter.get_meas_los(lo_dict)
        if m_los:
            dict_config['meas_lo_freq'] = m_los

    if user_lo_configs:
        # multiple frequency setups
        if len(qobj_schedules) == 1:
            # frequency sweep
            for lo_dict in user_lo_configs:
                experiments.append(PulseQobjExperiment(
                    instructions=qobj_schedules[0]['instructions'],
                    experimentheader=qobj_schedules[0]['header'],
                    experimentconfig=_lo_converter(lo_dict)
                ))
        elif len(qobj_schedules) == len(user_lo_configs):
            # n:n setup
            for lo_dict, schedule in zip(user_lo_configs, qobj_schedules):
                experiments.append(PulseQobjExperiment(
                    instructions=schedule['instructions'],
                    experimentheader=schedule['header'],
                    experimentconfig=_lo_converter(lo_dict)
                ))
        else:
            raise QiskitError('Invalid LO setting is specified. ' +
                              'This should be provided for each schedule, or ' +
                              'single setup for all schedules (unique), or ' +
                              'multiple setups for a single schedule (frequency sweep).')
    else:
        # unique frequency setup
        for schedule in qobj_schedules:
            experiments.append(PulseQobjExperiment(
                instructions=schedule['instructions'],
                experimentheader=schedule['header'],
            ))

    qobj_config = PulseQobjConfig(**dict_config)
    qobj_header = QobjHeader(**dict_header)

    return PulseQobj(qobj_id=str(uuid.uuid4()),
                     config=qobj_config,
                     experiments=experiments,
                     header=qobj_header)
