# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Assemble function for converting a list of circuits into a qobj"""
import uuid
import sympy
import numpy

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.qobj import Qobj, QobjConfig, QobjExperiment, QobjInstruction, QobjHeader
from qiskit.qobj import QobjExperimentConfig, QobjExperimentHeader, QobjConditional
from qiskit.compiler.run_config import RunConfig


def assemble_qobj(circuits, user_qobj_header=None, run_config=None, qobj_id=None):
    """Assembles a list of circuits into a qobj which can be run on the backend.

    Args:
        circuits (list[QuantumCircuits] or QuantumCircuit): circuits to compile
        user_qobj_header (QobjHeader): header to pass to the results
        run_config (RunConfig): RunConfig object
        qobj_id (int): identifier for the generated qobj

    Returns:
        Qobj: the Qobj to be run on the backends
    """
    user_qobj_header = user_qobj_header or QobjHeader()
    run_config = run_config or RunConfig()
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]

    userconfig = QobjConfig(**run_config.to_dict())
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
        experimentconfig = QobjExperimentConfig(n_qubits=n_qubits, memory_slots=memory_slots)

        instructions = []
        for opt in circuit.data:
            current_instruction = QobjInstruction(name=opt.name)
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
            # TODO: I really dont like this for snapshot. I also think we should change
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
        experiments.append(QobjExperiment(instructions=instructions, header=experimentheader,
                                          config=experimentconfig))
        if n_qubits > max_n_qubits:
            max_n_qubits = n_qubits
        if memory_slots > max_memory_slots:
            max_memory_slots = memory_slots

    userconfig.memory_slots = max_memory_slots
    userconfig.n_qubits = max_n_qubits

    return Qobj(qobj_id=qobj_id or str(uuid.uuid4()), config=userconfig,
                experiments=experiments, header=user_qobj_header)
